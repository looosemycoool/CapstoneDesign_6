import os
import re
from collections import defaultdict
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_upstage import UpstageEmbeddings

# BGE-m3-ko backend (lazy load, env=bge 일 때만 import 시도)
_BGE_MODEL = None
def _get_bge_model():
    global _BGE_MODEL
    if _BGE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _BGE_MODEL = SentenceTransformer("dragonkue/bge-m3-ko")
    return _BGE_MODEL

# ── 경로 / 환경변수 ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# 임베딩 backend 선택 — env EMBED_BACKEND={upstage|bge|contextual} (기본 upstage).
# upstage: collection knu_cse_upstage_pro, embedding-passage/query (API)
# bge: collection knu_cse_bge_m3_ko, dragonkue/bge-m3-ko (local, AutoRAG-ko 0.7456)
#   → 우리 도메인엔 -6 효과로 폐기 (Phase 7b)
# contextual: collection knu_cse_upstage_contextual, Anthropic Contextual Retrieval
#   chunk 앞에 LLM 생성 doc 컨텍스트 prepend → 검색 정확도 -67% 실패율 보고
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "upstage")
if EMBED_BACKEND == "bge":
    EXPERIMENT_ID = "bge_m3_ko"
    COLLECTION_NAME = "knu_cse_bge_m3_ko"
elif EMBED_BACKEND == "contextual":
    EXPERIMENT_ID = "upstage_contextual"
    COLLECTION_NAME = "knu_cse_upstage_contextual"
else:
    EXPERIMENT_ID = "upstage_pro"
    COLLECTION_NAME = f"knu_cse_{EXPERIMENT_ID}"

upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")


def get_embedding_function():
    """문서 인덱싱용 (vector DB 빌드 시 사용). Upstage backend 만 사용
    (BGE 는 02b_vector_db_bge.py 에서 sentence-transformers 직접 호출)."""
    return UpstageEmbeddings(
        model="embedding-passage",
        api_key=UPSTAGE_API_KEY,
    )


def _bge_embed_query(query: str) -> list[float]:
    """BGE-m3-ko 로 query 임베딩. normalize_embeddings=True 로 cosine 호환."""
    model = _get_bge_model()
    emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return emb.tolist()


class _BGEQueryEmbedder:
    """langchain Embeddings 같은 인터페이스만 흉내내는 wrapper.
    vector_search 가 .embed_query() 호출하므로 그것만 구현."""
    def embed_query(self, q): return _bge_embed_query(q)
    def embed_documents(self, qs): return [_bge_embed_query(q) for q in qs]


def get_query_embedding_function():
    """쿼리 임베딩 — EMBED_BACKEND 따라 Upstage embedding-query (API) 또는
    BGE-m3-ko (local sentence-transformers) 반환.
    Upstage 는 query 와 passage 에 별도 모델 권장 (정밀도 ↑)."""
    if EMBED_BACKEND == "bge":
        return _BGEQueryEmbedder()
    return UpstageEmbeddings(
        model="embedding-query",
        api_key=UPSTAGE_API_KEY,
    )


def get_graph_query_embedding_function():
    """Graph node search 전용 임베딩 — graph_nodes / graph_relations
    컬렉션이 항상 Upstage embedding-passage 로 빌드됐으므로 BGE backend
    에서도 query 는 Upstage embedding-query 사용해야 같은 공간 매칭됨.
    (Phase 6 에서 graph 가 빈 결과 반환한 원인 fix.)"""
    return UpstageEmbeddings(
        model="embedding-query",
        api_key=UPSTAGE_API_KEY,
    )


def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )


# ── Vector 검색 ──────────────────────────────────────────
RERANK_MODEL = "solar-pro2"  # cross-encoder 대안 — torch 없이 LLM rerank.
USE_RERANK = os.getenv("USE_RERANK", "1") == "1"
USE_BM25_HYBRID = os.getenv("USE_BM25_HYBRID", "0") == "1"  # Anthropic Contextual
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))  # rerank 전 1차 dense 검색량
BM25_K = int(os.getenv("BM25_K", "10"))            # BM25 1차 검색량
RERANK_TO_K = int(os.getenv("RERANK_TO_K", "5"))  # rerank 후 최종

# Phase 13 가설: rerank 가 vector 정확도 깎지만 (V↓6), graph anchor 의 시작점
# 으로는 도움 (더 topically relevant chunks). 분리해서 vector 답변엔 raw,
# graph anchor 엔 reranked 사용.
# USE_RERANK=0 + RERANK_FOR_GRAPH_ANCHOR=1 → vector raw + graph anchor reranked.
RERANK_FOR_GRAPH_ANCHOR = os.getenv("RERANK_FOR_GRAPH_ANCHOR", "0") == "1"

# BM25 캐시 (lazy build) — Chroma 전체 dump 후 코퍼스 토큰화. 첫 query 만 비용.
_BM25_CACHE = {"index": None, "docs": None}


def _build_bm25_index():
    """Chroma 전체 chunk 를 한 번만 dump 해서 BM25 코퍼스 구축.
    한국어 토큰화는 단순 whitespace split — 행정 문서는 띄어쓰기 단위로 핵심
    단어 (학번/금액/제도명) 가 분리돼 있어 충분. 형태소 분석기 (Mecab 등)
    추가 효과 작고 의존성 큼."""
    from rank_bm25 import BM25Okapi
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas"])
    docs = []
    tokenized = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        docs.append({
            "content": doc,
            "source": (meta or {}).get("file_name", ""),
        })
        # 토큰화: 공백 split + lowercase. 한국어/영어 혼재 대응.
        tokenized.append(doc.lower().split())
    _BM25_CACHE["index"] = BM25Okapi(tokenized)
    _BM25_CACHE["docs"] = docs


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 lexical retrieval. 한국어 행정 문서의 정확 토큰 (학번/금액/약어)
    매칭에 dense 보다 강함. dense 가 의미적 유사성을 잡는다면 BM25 는 정확
    매칭 (학번 '2023', 금액 '50만원' 등) 을 잡음.

    Anthropic Contextual Retrieval (2024-09): dense + BM25 RRF 만으로 retrieval
    실패율 -49%. cross-encoder rerank 추가 시 -67%."""
    if _BM25_CACHE["index"] is None:
        _build_bm25_index()
    tokens = query.lower().split()
    scores = _BM25_CACHE["index"].get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    out = []
    for i in top_idx:
        d = _BM25_CACHE["docs"][i]
        out.append({
            "content": d["content"],
            "source": d["source"],
            "score": float(scores[i]),
        })
    return out


def rrf_combine(*ranked_lists, k: int = 60, top_k: int = 10) -> list[dict]:
    """Reciprocal Rank Fusion — 여러 ranking 결합. 각 ranking 의 rank r 에 대해
    score = 1 / (k + r). 이론: TREC RRF (Cormack 2009). k=60 표준값.
    같은 doc 이 여러 ranking 에 나오면 score 합산.
    중복 판정은 (source, content[:100]) 키로."""
    fused_scores = {}
    fused_docs = {}
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking):
            key = (doc.get("source", ""), (doc.get("content", "") or "")[:100])
            if key not in fused_docs:
                fused_docs[key] = doc
                fused_scores[key] = 0.0
            fused_scores[key] += 1.0 / (k + rank + 1)
    sorted_keys = sorted(fused_scores.keys(), key=lambda k_: -fused_scores[k_])
    return [fused_docs[k_] for k_ in sorted_keys[:top_k]]


def _raw_vector_search(query, n_results):
    """Chroma 1차 dense retrieval (rerank 적용 안 함)."""
    embedding_fn = get_query_embedding_function()
    client = get_chroma_client()
    existing_collections = list(client.list_collections())
    if COLLECTION_NAME not in existing_collections:
        raise ValueError(
            f"컬렉션이 없습니다: {COLLECTION_NAME} | 현재 컬렉션: {existing_collections}"
        )
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = embedding_fn.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    docs = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(documents, metadatas, distances):
        score = round(1 - dist, 3) if dist is not None else None
        docs.append({
            "content": doc,
            "source": meta.get("file_name", "") if meta else "",
            "score": score,
        })
    return docs


def llm_rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """LLM (solar-pro2) cross-encoder 대안. top-K 후보를 query 관련성으로 재정렬.

    BGE-reranker-v2-m3 같은 SOTA reranker 가 가장 효과적이지만 torch 의존성
    무겁고 GPU 권장. 우리는 이미 사용 중인 solar-pro2 로 대체:
    - 모든 후보를 하나의 prompt 에 넣고 [순위 인덱스 list] 출력 요청
    - 1 query × 1 LLM call → +1.5초/query, 110 query 면 +3분 added

    Anthropic Contextual Retrieval (2024-09) 보고: rerank 만으로 retrieval 실패율
    -49% → -67%. cross-encoder 가 이상적이지만 LLM rerank 도 +5~10%p 보고됨.
    """
    if not docs or len(docs) <= top_k:
        return docs[:top_k]

    candidates_text = "\n".join(
        f"[{i}] (출처: {d.get('source','')[:40]})\n{d.get('content','')[:500]}"
        for i, d in enumerate(docs)
    )
    prompt = f"""다음 후보 문서 {len(docs)}개를 질문과의 관련성 높은 순서로 정렬하세요.

질문: {query}

후보:
{candidates_text}

규칙:
- 질문에 직접 답변할 수 있는 정보를 가진 문서가 상위
- 출처 파일명도 단서 (관련 주제일 가능성)
- 응답 형식: 가장 관련 높은 순서대로 인덱스 번호만 쉼표로 구분
- 예: 3, 0, 5, 7, 2 (총 {len(docs)}개 모두 포함)

순위:"""
    try:
        r = upstage_client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        text = (r.choices[0].message.content or "").strip()
        # parse "3, 0, 5, ..." → 인덱스 list
        import re as _re
        indices = []
        for tok in _re.split(r"[,，\s]+", text):
            tok = tok.strip().strip(".]:)[")
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(docs) and idx not in indices:
                    indices.append(idx)
        # 누락 인덱스는 원래 순서로 끝에 붙임 (안전망)
        for i in range(len(docs)):
            if i not in indices:
                indices.append(i)
        reranked = [docs[i] for i in indices[:top_k]]
        return reranked
    except Exception:
        # rerank 실패 시 원래 dense 순서 top-K
        return docs[:top_k]


def vector_search(query, n_results=5):
    """Retrieval pipeline:
      1. Dense (Chroma) RETRIEVE_K = 10
      2. (옵션) BM25 BM25_K = 10 → RRF 결합 → 상위 RETRIEVE_K
      3. (옵션) LLM rerank → 최종 RERANK_TO_K = 5

    토글 (env var):
      USE_RERANK=1 (기본)        — LLM rerank 적용
      USE_BM25_HYBRID=0 (기본)  — BM25 추가 + RRF
      RETRIEVE_K, BM25_K, RERANK_TO_K — 각 단계 K 조절

    Anthropic Contextual Retrieval (2024-09):
      dense only          baseline
      dense + BM25 + RRF  retrieval 실패율 -49%
      + cross-encoder     retrieval 실패율 -67%
    """
    dense = _raw_vector_search(query, n_results=RETRIEVE_K)

    if USE_BM25_HYBRID:
        try:
            sparse = bm25_search(query, top_k=BM25_K)
            candidates = rrf_combine(dense, sparse, top_k=RETRIEVE_K)
        except Exception:
            candidates = dense
    else:
        candidates = dense

    if USE_RERANK:
        return llm_rerank(query, candidates, top_k=RERANK_TO_K)
    else:
        return candidates[:n_results]


# ── Graph 검색 ───────────────────────────────────────────
# 의미 관계 화이트리스트 — MENTIONS 같은 메타 관계 제외하고 실제 의미를
# 가진 관계만 graph traversal 에 포함. (MENTIONS 는 전체 관계의 55.8% 차지하나
# Document → Entity 자동 메타라 hybrid 답변에 노이즈로 작용.)
MEANINGFUL_RELATIONS = [
    "REQUIRES", "HAS_CONDITION", "HAS_THRESHOLD", "HAS_DEADLINE",
    "HAS_EXCEPTION", "SUBSTITUTES_FOR", "ALTERNATIVE_PATH",
    "EXCLUDES_FROM", "TARGETS",
    "OFFERS", "PROVIDES", "INCLUDES", "PART_OF", "BELONGS_TO",
    "APPLIES_TO", "EXCLUDES", "RELATED_TO", "CHARGES",
    "REWARDS", "ACCEPTS", "REFERS",
]

# 조건/대체/예외 판정에 결정적인 관계 타입에 가중치 부여.
# 신규 lateral 관계(SUBSTITUTES_FOR, HAS_THRESHOLD, HAS_EXCEPTION, ALTERNATIVE_PATH)
# 는 정답에 직결되는 추론 단서라 1.6~1.8 로 가장 높게 부여.
RELATION_TYPE_WEIGHT = {
    "SUBSTITUTES_FOR": 1.8, "HAS_THRESHOLD": 1.7, "HAS_EXCEPTION": 1.7,
    "ALTERNATIVE_PATH": 1.6, "EXCLUDES_FROM": 1.6,
    "REQUIRES": 1.5, "HAS_CONDITION": 1.5,
    "TARGETS": 1.3, "HAS_DEADLINE": 1.3,
    "OFFERS": 1.2, "PROVIDES": 1.2, "INCLUDES": 1.2,
    "EXCLUDES": 1.2, "APPLIES_TO": 1.2,
    "PART_OF": 1.0, "BELONGS_TO": 1.0,
    "RELATED_TO": 0.8, "CHARGES": 1.1,
    "REWARDS": 1.0, "ACCEPTS": 1.0, "REFERS": 0.8,
}


GRAPH_NODES_COLLECTION = "knu_cse_graph_nodes"  # build_graph_embeddings.py 가 생성
GRAPH_RELATIONS_COLLECTION = "knu_cse_graph_relations"  # build_relation_embeddings.py 가 생성

# v3 toggle: LightRAG dual-seed + doc-restricted 2-hop + PathRAG flow scoring + CRAG safety.
# 환경변수 HYBRID_V3=1 로 활성화 (eval 돌릴 때 코드 수정 없이 분기 비교용).
# 기본값 False → 검증된 임베딩 기반 1-hop (+1).
USE_GRAPH_SEARCH_V3 = os.getenv("HYBRID_V3", "0") == "1"
HIGH_LEVEL_KW_MODEL = "solar-pro2"  # high-level keyword 추출 (cheap, fast)


def graph_search(query, vector_docs=None, max_relations=5, n_seed_nodes=8):
    """Embedding-based semantic GraphRAG.

    Phase 1 재추출 후 그래프 노드명이 한국어 통일되고 lateral 관계
    (SUBSTITUTES_FOR, HAS_THRESHOLD, HAS_EXCEPTION, ALTERNATIVE_PATH,
    EXCLUDES_FROM) 가 추가됨. 정통 풀버전 (doc-restricted query-anchored)
    이 오히려 V=64 H=58 (-6) 으로 후퇴해서 검증된 임베딩 기반 (+1) 으로
    복귀. 그래프 데이터 품질 개선이 retrieval 보다 효과 큼.

    Pipeline:
    1. 사전 임베딩된 graph 노드 (knu_cse_graph_nodes Chroma collection)
       에서 query 와 의미 유사 top-k seed 선택
    2. 각 seed 에서 1-hop 의미 관계 (MEANINGFUL_RELATIONS) expansion
    3. Score = seed similarity × relation type weight × subgraph cohesion
              × hub penalty

    vector_docs 는 호환성을 위해 시그니처에 유지 (사용 안 함).
    """
    # graph_nodes collection 은 항상 Upstage 임베딩으로 빌드됐으므로
    # query 임베딩도 같은 공간 (Upstage embedding-query) 사용해야 매칭됨.
    embedding_fn = get_graph_query_embedding_function()
    chroma = get_chroma_client()

    existing = list(chroma.list_collections())
    if GRAPH_NODES_COLLECTION not in existing:
        return []

    coll = chroma.get_collection(GRAPH_NODES_COLLECTION)

    q_emb = embedding_fn.embed_query(query)
    res = coll.query(query_embeddings=[q_emb], n_results=n_seed_nodes)
    seed_ids = res.get("ids", [[]])[0]
    seed_metas = res.get("metadatas", [[]])[0]
    seed_distances = res.get("distances", [[]])[0]

    if not seed_ids:
        return []

    seeds = []  # [(elementId, name, similarity)]
    for sid, meta, dist in zip(seed_ids, seed_metas, seed_distances):
        sim = 1.0 / (1.0 + max(0.0, dist or 0))
        seeds.append((sid, meta.get("name", ""), sim))

    seed_names = {s[1] for s in seeds if s[1]}

    driver = get_neo4j_driver()
    seen = {}

    try:
        with driver.session() as session:
            for seed_id, seed_name, sim in seeds:
                rows = session.run("""
                    MATCH (a)-[r]-(b)
                    WHERE elementId(a) = $id
                      AND NOT b:Document
                      AND type(r) IN $types
                    RETURN
                        coalesce(startNode(r).name, '') AS from_name,
                        type(r) AS rel,
                        coalesce(endNode(r).name, '') AS to_name,
                        coalesce(b.name, '') AS neighbor_name,
                        COUNT { (b)--() } AS neighbor_degree
                    LIMIT 15
                """, {"id": seed_id, "types": MEANINGFUL_RELATIONS}).data()

                for row in rows:
                    from_n = (row.get("from_name") or "").strip()
                    to_n = (row.get("to_name") or "").strip()
                    rel = (row.get("rel") or "").strip()
                    neighbor = (row.get("neighbor_name") or "").strip()
                    neighbor_deg = row.get("neighbor_degree") or 1

                    if not from_n or not to_n or not rel:
                        continue

                    score = sim
                    score *= RELATION_TYPE_WEIGHT.get(rel, 1.0)

                    if neighbor in seed_names:
                        score *= 1.5

                    if neighbor_deg > 20:
                        score *= 0.5
                    elif neighbor_deg > 10:
                        score *= 0.7

                    key = (from_n, rel, to_n)
                    if key not in seen or seen[key]["score"] < score:
                        seen[key] = {
                            "from": from_n, "relation": rel, "to": to_n,
                            "score": score,
                        }
    finally:
        driver.close()

    triples = sorted(seen.values(), key=lambda x: (-x["score"], x["from"]))
    return [
        {"from": t["from"], "relation": t["relation"], "to": t["to"]}
        for t in triples[:max_relations]
    ]


# ═══════════════════════════════════════════════════════════════════
# Vector-anchored graph search (Phase 10) — body-grounded only
# ═══════════════════════════════════════════════════════════════════
# 동기 (Phase 8/9b 결과): query/embedding seed 기반 graph_search 가 vector
# 본문과 disconnect 된 triple 을 가져와서 LLM 답변을 망쳤음 (V=73 → H=64).
# 해결: vector top-K 본문에 *명시적으로 등장* 하는 entity 만 graph 시작점으로
# 사용하고, 1-hop 결과 중 *양쪽 endpoint 가 모두 본문에 등장* 하는 triple
# 만 keep. 이러면 graph 가 "본문 안에서 발견된 entity 들의 명시적 관계"
# 만 표면화 — 노이즈 0 보장.

# 모든 graph 노드 이름 캐시 (1번만 빌드).
_GRAPH_NODE_NAMES = None


def _get_all_graph_node_names() -> list[str]:
    global _GRAPH_NODE_NAMES
    if _GRAPH_NODE_NAMES is None:
        driver = get_neo4j_driver()
        names = []
        try:
            with driver.session() as s:
                rows = s.run(
                    "MATCH (n) WHERE NOT n:Document AND coalesce(n.name,'') <> '' "
                    "RETURN n.name AS name"
                ).data()
                names = [(r["name"] or "").strip() for r in rows if r["name"]]
        finally:
            driver.close()
        # 길이 ≥ 2 만 (1글자 false-positive 회피)
        _GRAPH_NODE_NAMES = sorted({n for n in names if len(n) >= 2}, key=lambda x: -len(x))
    return _GRAPH_NODE_NAMES


def _entities_in_body(body_text: str) -> set[str]:
    """본문 텍스트에서 graph 노드 이름이 substring 으로 등장하는 것만 추출.
    긴 entity 우선 매칭 (한국어 행정 도메인의 합성어 매칭에 유리)."""
    if not body_text:
        return set()
    found = set()
    for name in _get_all_graph_node_names():
        if name in body_text:
            found.add(name)
    return found


def vector_anchored_graph_search(
    query, vector_docs, max_relations: int = 5,
) -> list[dict]:
    """진짜 vector-anchored — vector 본문에 등장한 entity 만 graph 시작점으로,
    1-hop 결과 중 양쪽 endpoint 모두 본문에 등장한 triple 만 surface.

    Phase 8/9b 의 노이즈 ("query embedding 으로 graph node 잡는 방식이 본문과
    disconnect") 를 완전 제거. graph 가 본문에 없는 정보를 끌어올 수 없음.
    """
    if not vector_docs:
        return []
    body = "\n".join((d.get("content") or "") for d in vector_docs)
    body_entities = _entities_in_body(body)
    if len(body_entities) < 2:
        # entity 1개 이하면 의미있는 관계 형성 불가
        return []

    driver = get_neo4j_driver()
    seen = {}
    try:
        with driver.session() as s:
            rows = s.run("""
                MATCH (a)-[r]->(b)
                WHERE NOT a:Document AND NOT b:Document
                  AND coalesce(a.name,'') IN $names
                  AND coalesce(b.name,'') IN $names
                  AND type(r) IN $types
                RETURN a.name AS f, type(r) AS rel, b.name AS t
            """, {
                "names": list(body_entities),
                "types": MEANINGFUL_RELATIONS,
            }).data()
        for row in rows:
            f, rel, t = row["f"].strip(), row["rel"].strip(), row["t"].strip()
            if not f or not t or not rel or f == t:
                continue
            score = RELATION_TYPE_WEIGHT.get(rel, 1.0)
            key = (f, rel, t)
            if key not in seen or seen[key]["score"] < score:
                seen[key] = {"from": f, "relation": rel, "to": t, "score": score}
    finally:
        driver.close()

    out = sorted(seen.values(), key=lambda x: -x["score"])
    return [{"from": t["from"], "relation": t["relation"], "to": t["to"]}
            for t in out[:max_relations]]


# ═══════════════════════════════════════════════════════════════════
# Plan B v2: LightRAG dual-seed + doc-restricted 2-hop + PathRAG flow
# ═══════════════════════════════════════════════════════════════════
# 출처:
# - LightRAG (arxiv 2410.05779): low-level (entity) + high-level (theme/relation)
#   키워드를 LLM 으로 추출해 두 개의 vector index 에 매칭.
# - PathRAG (arxiv 2502.14902): flow-based pruning (resource propagation) +
#   reliability 오름차순 prompt ordering (recency bias). Ablation 결과 path
#   prompting 이 flat triple dump 대비 +56% win rate.
# - CRAG (arxiv 2401.15884): retrieval evaluator → 신뢰도 낮으면 graph skip.

def extract_high_level_keywords(query: str, max_keywords: int = 5) -> list[str]:
    """LightRAG dual-level 의 high-level keyword 추출.
    질문의 *관계 의도* (조건/예외/대체/마감 등) 를 뽑아 relation index 와 매칭.

    예: "복학생도 계절학기 신청 가능?" → ["조건", "대상", "예외"]
        "글솝 졸업요건 중 기술창업역량은?" → ["요건", "조건", "포함"]
    """
    prompt = f"""아래 질문이 묻는 *관계 의도* 를 한국어 명사 키워드로 뽑으세요.

규칙:
- entity (군휴학, 글솝 등 고유명사) 가 아니라 *어떤 관계를 묻는지* 의 키워드만
- 예시 카테고리: 조건, 자격, 대상, 요건, 마감, 예외, 대체, 동등경로, 제외, 포함, 적용, 비용, 보상, 임계값
- 쉼표로만 구분, 추가 설명 금지, 최대 {max_keywords}개

질문: {query}
관계 키워드:"""
    try:
        r = upstage_client.chat.completions.create(
            model=HIGH_LEVEL_KW_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        text = (r.choices[0].message.content or "").strip()
        kws = [k.strip().strip("·-•*").strip() for k in re.split(r"[,，、\n]+", text)]
        kws = [k for k in kws if 2 <= len(k) <= 20]
        return kws[:max_keywords]
    except Exception:
        return []


def _entity_seed_search(query: str, top_n: int = 8) -> list[dict]:
    """현재 graph_search 와 같은 방식 — query 임베딩으로 entity 노드 top-N.
    반환: [{"node_id", "name", "sim"}, ...] (sim = 1/(1+dist))"""
    chroma = get_chroma_client()
    if GRAPH_NODES_COLLECTION not in list(chroma.list_collections()):
        return []
    coll = chroma.get_collection(GRAPH_NODES_COLLECTION)
    # graph_nodes 는 Upstage 임베딩으로 빌드 → 같은 공간 매칭 필요
    embedding_fn = get_graph_query_embedding_function()
    q_emb = embedding_fn.embed_query(query)
    res = coll.query(query_embeddings=[q_emb], n_results=top_n)
    out = []
    for sid, meta, dist in zip(
        res.get("ids", [[]])[0],
        res.get("metadatas", [[]])[0],
        res.get("distances", [[]])[0],
    ):
        sim = 1.0 / (1.0 + max(0.0, dist or 0))
        out.append({"node_id": sid, "name": (meta or {}).get("name", ""), "sim": sim})
    return out


def _relation_seed_search(high_kws: list[str], top_m: int = 6) -> list[dict]:
    """LightRAG high-level 매칭 — keyword 임베딩으로 relation index top-M.
    매칭된 relation 의 양쪽 노드명 반환 (graph 시작점 후보).
    반환: [{"name", "sim"}, ...]"""
    if not high_kws:
        return []
    chroma = get_chroma_client()
    if GRAPH_RELATIONS_COLLECTION not in list(chroma.list_collections()):
        return []
    coll = chroma.get_collection(GRAPH_RELATIONS_COLLECTION)
    # graph_relations 도 Upstage 임베딩으로 빌드 → 같은 공간 매칭 필요
    embedding_fn = get_graph_query_embedding_function()
    # 각 키워드별 검색 후 합치는 방식 — 단일 임베딩으로 쳐도 되지만 키워드별
    # 검색이 facet 분리에 더 유리.
    seed_nodes = {}  # name -> max sim
    per_kw_top = max(2, top_m // max(len(high_kws), 1) + 1)
    for kw in high_kws:
        try:
            kw_emb = embedding_fn.embed_query(kw)
            res = coll.query(query_embeddings=[kw_emb], n_results=per_kw_top)
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for meta, dist in zip(metas, dists):
                sim = 1.0 / (1.0 + max(0.0, dist or 0))
                for name in (meta.get("from_name", ""), meta.get("to_name", "")):
                    name = name.strip()
                    if not name:
                        continue
                    if name not in seed_nodes or seed_nodes[name] < sim:
                        seed_nodes[name] = sim
        except Exception:
            continue
    out = sorted(
        [{"name": n, "sim": s} for n, s in seed_nodes.items()],
        key=lambda x: -x["sim"],
    )
    return out[:top_m]


def _fuzzy_file_match(graph_files: list, vector_files: set) -> bool:
    """vector file_name 과 graph source_files 형식 차이 (notice 는 [공지] |
    suffix) 우회용 fuzzy 매칭."""
    for gf in graph_files or []:
        for vf in vector_files:
            if gf == vf or vf in gf or gf in vf:
                return True
    return False


def _fetch_paths_from_seeds(
    seed_node_ids: list[str], seed_node_names: list[str],
    files: set, max_hops: int = 2, limit: int = 200,
) -> list[dict]:
    """Doc 제한 안에서 seed 들로부터 1~max_hops path 추출.
    seed_node_ids (elementId, entity 임베딩 결과) + seed_node_names
    (relation index 매칭 결과) 둘 다 시작점으로.
    files 비면 doc 제한 미적용."""
    if not seed_node_ids and not seed_node_names:
        return []

    driver = get_neo4j_driver()
    paths_out = []
    try:
        with driver.session() as session:
            cypher = f"""
                MATCH (start)
                WHERE NOT start:Document
                  AND coalesce(start.name, '') <> ''
                  AND (
                       elementId(start) IN $seed_ids
                       OR start.name IN $seed_names
                  )
                WITH DISTINCT start
                MATCH path = (start)-[r*1..{max_hops}]-(neighbor)
                WHERE ALL(rel IN r WHERE type(rel) IN $types)
                  AND NOT neighbor:Document
                  AND coalesce(neighbor.name, '') <> ''
                RETURN
                    [n IN nodes(path) | coalesce(n.name, '')] AS path_nodes,
                    [rel IN relationships(path) | type(rel)] AS path_rels,
                    length(path) AS hop_len,
                    [n IN nodes(path) | coalesce(n.source_files, [])] AS node_files
                ORDER BY hop_len ASC
                LIMIT $limit
            """
            rows = session.run(cypher, {
                "seed_ids": seed_node_ids,
                "seed_names": seed_node_names,
                "types": MEANINGFUL_RELATIONS,
                "limit": limit,
            }).data()
        for row in rows:
            nodes = [n for n in (row.get("path_nodes") or []) if n]
            rels = row.get("path_rels") or []
            if len(nodes) < 2 or not rels:
                continue
            node_files = row.get("node_files") or []
            # Doc 제한: vector files 있을 때만 in-doc 체크
            if files:
                in_doc = all(_fuzzy_file_match(nfs, files) for nfs in node_files)
            else:
                in_doc = True
            paths_out.append({
                "nodes": nodes,
                "relations": rels,
                "hop": row.get("hop_len") or 1,
                "in_doc": in_doc,
            })
    finally:
        driver.close()
    return paths_out


def _flow_propagate_reliability(
    paths: list[dict], seed_sim_map: dict, alpha: float = 0.7,
) -> list[dict]:
    """PathRAG flow-based pruning 의 단순화 버전.
    - 각 seed 노드의 초기 resource = 그 노드의 query similarity (entity 매칭 sim)
    - hop 마다 α 감쇠
    - path reliability = avg(node resources on path) × avg(relation type weight)
                        × (in_doc ? 1.5 : 1.0)
    """
    if not paths:
        return []
    # node → reliability 누적 (path 안 모든 노드의 resource 합산 평균)
    enriched = []
    for p in paths:
        nodes = p["nodes"]
        rels = p["relations"]
        # node resource: seed 면 sim, 아니면 hop 거리 기반 감쇠
        node_resources = []
        for i, n in enumerate(nodes):
            if n in seed_sim_map:
                node_resources.append(seed_sim_map[n])
            else:
                # 가장 가까운 seed 까지의 hop 추정 (간단히 전체 hop / 2)
                # 정확하진 않지만 path 내 위치 기반 감쇠라 OK
                hop_distance = min(i, len(nodes) - 1 - i) + 1
                node_resources.append(alpha ** hop_distance)
        avg_resource = sum(node_resources) / len(node_resources)
        avg_rel_w = sum(RELATION_TYPE_WEIGHT.get(r, 1.0) for r in rels) / len(rels)
        bridge = 1.5 if p.get("in_doc") else 1.0
        reliability = avg_resource * avg_rel_w * bridge
        enriched.append({**p, "reliability": reliability})
    return enriched


def _path_dedup_and_top(paths: list[dict], top_k: int) -> list[dict]:
    """Path dedup (방향 무시) 후 reliability 내림차순 top-K."""
    seen, out = set(), []
    for p in sorted(paths, key=lambda x: -x["reliability"]):
        nodes = p["nodes"]
        key = tuple(nodes) if nodes[0] <= nodes[-1] else tuple(reversed(nodes))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= top_k:
            break
    return out


def graph_search_v3(
    query, vector_docs=None, max_paths=5, max_hops=2,
    crag_threshold=0.15,
) -> list[dict]:
    """SOTA 합성판 — LightRAG dual seed + doc-restricted 2-hop + PathRAG
    flow scoring + CRAG safety.

    반환: paths = [{nodes:[...], relations:[...], hop, reliability}, ...]
    빈 배열은 CRAG 가 graph skip 결정한 경우 (모두 reliability < threshold).
    """
    # 1. dual seed
    entity_seeds = _entity_seed_search(query, top_n=8)
    high_kws = extract_high_level_keywords(query)
    relation_seeds = _relation_seed_search(high_kws, top_m=6)

    # seed 통합 — entity 는 elementId, relation 매칭은 name 기반
    seed_node_ids = [s["node_id"] for s in entity_seeds if s.get("node_id")]
    seed_node_names = list({s["name"] for s in relation_seeds if s.get("name")})
    # name → sim 매핑 (flow 계산용). entity 검색의 name 도 포함.
    seed_sim_map = {}
    for s in entity_seeds:
        if s.get("name"):
            seed_sim_map[s["name"]] = max(seed_sim_map.get(s["name"], 0), s["sim"])
    for s in relation_seeds:
        if s.get("name"):
            seed_sim_map[s["name"]] = max(seed_sim_map.get(s["name"], 0), s["sim"])

    if not seed_node_ids and not seed_node_names:
        return []

    # 2. doc 제한 파일 set
    files = set()
    if vector_docs:
        for d in vector_docs:
            f = (d.get("source") or "").strip()
            if f:
                files.add(f)

    # 3. 2-hop path retrieval
    raw_paths = _fetch_paths_from_seeds(
        seed_node_ids, seed_node_names, files, max_hops=max_hops, limit=200,
    )
    if not raw_paths:
        return []

    # 4. PathRAG flow scoring
    scored = _flow_propagate_reliability(raw_paths, seed_sim_map, alpha=0.7)

    # 5. dedup + top-K
    top_paths = _path_dedup_and_top(scored, top_k=max_paths)

    # 6. CRAG safety: 모두 reliability 낮으면 graph skip
    if not top_paths or top_paths[0]["reliability"] < crag_threshold:
        return []

    return top_paths


def paths_to_edges(paths: list[dict]) -> list[dict]:
    """Path 리스트를 평가/로깅용 단일 edge 리스트로 풀어냄.
    eval (summarize_graph_relations) 가 {from,relation,to} 포맷 기대.
    Reliability 가 path 단위라 같은 path 의 모든 edge 가 동일 score 받음."""
    edges = []
    seen = set()
    for p in paths:
        nodes = p.get("nodes") or []
        rels = p.get("relations") or []
        for i, rel in enumerate(rels):
            if i + 1 >= len(nodes):
                break
            f, t = nodes[i], nodes[i + 1]
            key = (f, rel, t)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": f, "relation": rel, "to": t})
    return edges


def _path_to_korean(path: dict) -> str:
    """단일 path → 자연어 한 줄 (PathRAG 스타일).
    1-hop: '복학 → 계절학기  (요구)'
    2-hop: '군휴학 → 복학 → 계절학기  (요구 / 적용)'"""
    nodes = path.get("nodes") or []
    rels = path.get("relations") or []
    if len(nodes) < 2 or not rels:
        return ""
    chain = " → ".join(nodes)
    rels_ko = [REL_KO_SHORT.get(r, r) for r in rels]
    return f"- {chain}  ({' / '.join(rels_ko)})"


REL_KO_SHORT = {
    "REQUIRES": "요구", "HAS_DEADLINE": "마감", "HAS_CONDITION": "조건",
    "HAS_THRESHOLD": "임계값", "HAS_EXCEPTION": "예외",
    "SUBSTITUTES_FOR": "대체", "ALTERNATIVE_PATH": "동등경로",
    "EXCLUDES_FROM": "제외대상", "TARGETS": "대상",
    "PROVIDES": "제공", "OFFERS": "제공", "INCLUDES": "포함",
    "REWARDS": "보상", "CHARGES": "비용", "PART_OF": "일부",
    "BELONGS_TO": "소속", "APPLIES_TO": "적용", "EXCLUDES": "제외",
    "RELATED_TO": "관련", "ACCEPTS": "수용", "REFERS": "참조",
}


# ── Graph 관계 타입을 한국어 술어로 변환 (LLM 가독성 ↑) ──────
REL_KO = {
    "REQUIRES":         "는 다음을 요구함:",
    "HAS_DEADLINE":     "의 마감일:",
    "HAS_CONDITION":    "의 조건:",
    "HAS_THRESHOLD":    "의 임계값:",
    "HAS_EXCEPTION":    "의 예외:",
    "SUBSTITUTES_FOR":  "는 다음을 대체함:",
    "ALTERNATIVE_PATH": "와 동등한 경로:",
    "EXCLUDES_FROM":    "는 다음에서 제외됨:",
    "TARGETS":          "의 대상:",
    "PROVIDES":         "는 다음을 제공함:",
    "OFFERS":           "는 다음을 제공:",
    "INCLUDES":         "는 다음을 포함:",
    "REWARDS":          "는 다음에 보상함:",
    "CHARGES":          "는 다음에 비용 부과:",
    "PART_OF":          "는 다음의 일부:",
    "BELONGS_TO":       "는 다음에 속함:",
    "APPLIES_TO":       "는 다음에 적용됨:",
    "EXCLUDES":         "는 다음을 제외:",
    "RELATED_TO":       "는 관련됨:",
    "MENTIONS":         "에서 언급됨:",
}


# ── Graph noise gating ──────────────────────────────────
# Failure analysis (46 H 오답 중 17건 = 37%) 결과: graph triple 이 본문과
# 무관한 다른 트랙/장학생/학번의 임계값/예외를 끌어와 LLM 이 그것을 답변에
# 인용. 예) 다중전공트랙 질문에 "졸업요건 HAS_THRESHOLD 토익 800점"
# (실제는 해외복수학위트랙 임계값) 주입 → 토익 700→800 오답.
#
# 해결: graph triple 을 prompt 에 포함하기 전, triple 의 entity 가 vector
# 검색된 본문 chunk 에 substring 으로 등장하는지 검증. 한쪽도 안 나오면 drop.
# HAS_THRESHOLD/EXCLUDES_FROM/EXCLUDES/HAS_EXCEPTION 같은 임계값/예외 관계는
# 더 엄격하게 양쪽 모두 등장 요구.
USE_GRAPH_GATING = os.getenv("USE_GRAPH_GATING", "0") == "1"  # Phase 5 결과
# inconclusive (V 에 29 pipeline error 섞임, H=42 신뢰 불가) → 일단 OFF.
# 향후 retest 가능 — 코드는 보존.
STRICT_GATING_RELS = {
    "HAS_THRESHOLD", "EXCLUDES", "EXCLUDES_FROM", "HAS_EXCEPTION",
    "SUBSTITUTES_FOR",  # 대체 관계도 잘못 적용되면 핵심 오답 유발
}


def _entity_in_body(entity: str, body_text: str, min_token_len: int = 2) -> bool:
    """entity 가 body 에 등장하는지 검사. 정확 substring 우선,
    부분 매칭은 entity 의 의미 토큰 (길이 ≥ min_token_len) 이 나오는지로 판정.
    예: '평점 1.7 미만 학생' → ['평점','1.7','미만','학생'] 중 하나 매칭."""
    if not entity:
        return False
    entity = entity.strip()
    if entity in body_text:
        return True
    # 부분 매칭: 의미 토큰 (조사 제외 일반 단어/숫자) 중 절반 이상 매칭
    tokens = [
        t for t in re.split(r"[\s/·,()\\[\\]\"']+", entity)
        if len(t) >= min_token_len
    ]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in body_text)
    return hits / len(tokens) >= 0.5


def gate_graph_by_body(graph_data: list[dict], vector_docs: list[dict]) -> list[dict]:
    """Graph triple/path 가 본문과 정합한지 검증해 노이즈 drop.

    rule:
    - 일반 관계: 양 끝 entity 중 *최소 1개* 가 본문에 등장 → keep
    - STRICT_GATING_RELS (HAS_THRESHOLD, EXCLUDES, EXCLUDES_FROM, HAS_EXCEPTION,
      SUBSTITUTES_FOR): 양 끝 entity *둘 다* 본문에 등장해야 keep
      (이들 관계가 실패 케이스 17건 중 다수의 원인)

    edge 형식 ({from,relation,to}) 와 path 형식 ({nodes,relations}) 둘 다 처리."""
    if not graph_data or not vector_docs:
        return graph_data

    body_text = "\n".join(
        (d.get("content") or "") for d in vector_docs
    )
    if not body_text.strip():
        return graph_data

    is_path_format = isinstance(graph_data[0], dict) and "nodes" in graph_data[0]
    filtered = []

    if is_path_format:
        for p in graph_data:
            nodes = p.get("nodes") or []
            rels = p.get("relations") or []
            if len(nodes) < 2:
                continue
            # path 의 모든 노드 검사 → strict 관계 포함 시 모두 등장 필요,
            # 아니면 한 개라도 등장하면 keep
            has_strict = any(r in STRICT_GATING_RELS for r in rels)
            if has_strict:
                if all(_entity_in_body(n, body_text) for n in nodes):
                    filtered.append(p)
            else:
                if any(_entity_in_body(n, body_text) for n in nodes):
                    filtered.append(p)
    else:
        for t in graph_data:
            from_n = t.get("from", "")
            to_n = t.get("to", "")
            rel = t.get("relation", "")
            from_hit = _entity_in_body(from_n, body_text)
            to_hit = _entity_in_body(to_n, body_text)
            if rel in STRICT_GATING_RELS:
                if from_hit and to_hit:
                    filtered.append(t)
            else:
                if from_hit or to_hit:
                    filtered.append(t)

    return filtered


# ── 결과 통합 ─────────────────────────────────────────────
def merge_results(vector_docs, graph_data):
    """Vector + Graph 결과를 컨텍스트 문자열로 통합.

    graph_data 가 path 형식 ({"nodes":[...], "relations":[...]}) 면 v3 의
    PathRAG 스타일 출력 (chain 자연어 + reliability 오름차순 → 가장 신뢰
    높은 path 가 prompt 끝, recency bias 활용). edge 형식이면 기존 단일
    triple 출력. graph 블록은 항상 vector 보다 앞 배치 (positional weight).
    """
    context_parts = []

    if graph_data:
        is_path_format = isinstance(graph_data[0], dict) and "nodes" in graph_data[0]

        graph_text = "=== 핵심 관계/조건 단서 (조건·자격·대체·예외 판정용) ===\n"

        if is_path_format:
            # PathRAG 스타일: reliability 오름차순 (마지막 = 가장 신뢰)
            ordered = sorted(graph_data, key=lambda p: p.get("reliability", 0))
            for p in ordered:
                line = _path_to_korean(p)
                if line:
                    graph_text += line + "\n"
        else:
            for rel in graph_data:
                if rel.get("from") and rel.get("to"):
                    verb = REL_KO.get(rel["relation"], f"--[{rel['relation']}]-->")
                    graph_text += f"- {rel['from']} {verb} {rel['to']}\n"

        if graph_text.strip().count("\n") >= 1:  # 헤더 외 한 줄이라도 있으면
            context_parts.append(graph_text.strip())

    if vector_docs:
        vector_text = "=== 문서 본문 (서술·배경 근거용) ===\n"
        for i, doc in enumerate(vector_docs, start=1):
            vector_text += f"\n[{i}] 출처: {doc['source']} (유사도: {doc['score']})\n"
            vector_text += f"{doc['content']}\n"
        context_parts.append(vector_text.strip())

    return "\n\n".join(context_parts).strip()


# ── LLM 답변 생성 ─────────────────────────────────────────
GENERATION_MODEL = "solar-pro3"  # 단일 source of truth — 호출부에서 이 상수 참조
GENERATION_TEMPERATURE = 0.2     # 논문 기록용 (낮은 값으로 결정성/재현성 우선)
GENERATION_MAX_TOKENS = None     # None=모델 기본값. Phase 2 ablation 결과 cap=400 이
                                 # vector(-5) hybrid(-6) 모두 떨어뜨림 → 무제한 복귀.

# Chain-of-Verification (Phase 11) toggle.
# Anchored mode 가 -4 까지 좁혔으나 H>V 못 만듦 (graph 의 marginal value=0).
# CoV: vector 답변 생성 → graph triple 로 LLM 검증/보강 → final answer.
# 출처: Dhuliawala et al. 2023 "Chain-of-Verification Reduces Hallucination"
USE_VERIFICATION = os.getenv("USE_VERIFICATION", "0") == "1"
VERIFICATION_MODEL = "solar-pro3"


def verify_and_refine(query: str, draft_answer: str,
                      vector_docs: list, graph_relations: list) -> str:
    """vector 답변 (draft) + graph triple 보고 LLM 이 검증/보강 → final.
    graph 단서가 draft 에 빠진 정답 사실 (특히 조건/예외/임계값) 알려주거나,
    draft 의 모순 fact 을 본문 기반으로 교정."""
    if not graph_relations:
        return draft_answer  # graph 없으면 검증 단계 스킵

    triples_text = "\n".join(
        f"- {r['from']} --[{r['relation']}]--> {r['to']}"
        for r in graph_relations[:10]
    )
    body_text = "\n".join(
        f"[{i+1}] {(d.get('content') or '')[:600]}"
        for i, d in enumerate(vector_docs[:5])
    )

    prompt = f"""당신은 답변 검증 전문가입니다. 아래 *초안 답변* 을 *문서 본문* 과
*지식그래프 단서* 에 비추어 검토하고, 필요시 보강하거나 교정한 *최종 답변* 을
출력하세요.

[질문]
{query}

[초안 답변]
{draft_answer}

[문서 본문]
{body_text}

[지식그래프 단서 (조건/관계 보조)]
{triples_text}

[검증 규칙]
1. 초안에 사실 오류가 있으면 본문 기반으로 교정.
2. 본문에 명시된 핵심 사실 (수치/날짜/예외/면제) 이 초안에 빠졌으면 추가.
3. 그래프 단서가 본문에 명시된 관계 (예: 자격/대체/예외) 라면 답변에 명시.
4. 그래프 단서가 본문에 없거나 모순되면 무시 (본문 우선).
5. 초안이 이미 정확하고 완전하면 그대로 출력 (불필요한 변형 금지).

[최종 답변]
"""
    try:
        r = upstage_client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=GENERATION_TEMPERATURE,
        )
        refined = (r.choices[0].message.content or "").strip()
        return refined if refined else draft_answer
    except Exception:
        return draft_answer


def generate_answer(query, context, model=GENERATION_MODEL):
    """컨텍스트 기반 LLM 답변 생성. 모델/온도/max_tokens 는 모듈 상수 참조."""
    if not context.strip():
        return "해당 정보를 찾을 수 없습니다."
    prompt = f"""당신은 경북대학교 컴퓨터학부 학생들을 위한 AI 챗봇입니다.

아래 검색 결과를 바탕으로 질문에 답변하세요. 검색 결과는 두 가지로 구성됩니다:
- [핵심 관계/조건 단서]: 지식 그래프에서 추출한 개체 간 관계 — 본문 검증/보강 *보조 자료*. 본문에 같은 정보가 있으면 본문 표현 사용. 본문에 없거나 불충분한 부분만 단서로 보강.
- [문서 본문]: 공지/문서의 실제 서술. **답변의 1차 근거** — 사실, 수치, 자격, 조건, 예외, 면제 모두 본문 우선.

답변 규칙:
1. **본문 우선** — 단서와 본문이 충돌하면 본문 채택. 단서만 있고 본문에 없으면 단서로 답하기 전 정말 질문과 일치하는지 다시 확인.
2. 검색 결과에 없는 내용은 추측 금지 — "해당 정보를 찾을 수 없습니다." 답변.
3. 본문에 "면제", "예외", "단", "다만" 같은 한정/예외 조항이 있으면 절대 누락 금지.

[검색 결과]
{context}

[질문]
{query}

[답변]
"""

    api_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": GENERATION_TEMPERATURE,
    }
    # 상수가 None 이 아니면 실제로 cap 적용 (None 일 땐 모델 기본값).
    # CodeRabbit 지적: 상수만 정의하고 호출부에서 사용 안 하면 paper 와 코드가
    # 어긋나 재현성이 깨진다.
    if GENERATION_MAX_TOKENS is not None:
        api_kwargs["max_tokens"] = GENERATION_MAX_TOKENS

    response = upstage_client.chat.completions.create(**api_kwargs)

    return response.choices[0].message.content.strip()


# Graph 활용 모드 — env GRAPH_USAGE={anchored|context|expansion|off}
# anchored (Phase 10, 신규 default): vector_anchored_graph_search 사용.
#   Vector 본문에 등장한 entity 만 graph 시작점, 양쪽 endpoint 본문에
#   등장한 triple 만 surface → 노이즈 0. LLM context 에 주입.
# context (legacy): query embedding seed graph_search → triple 을 LLM 주입
#   (Phase 8 결과 -9 regression — anti-pattern in admin docs)
# expansion: graph entity 로 query 확장 → vector 재검색 (Phase 9b 결과 -9)
# off: graph 완전 비활성 (= vector_only_rag 와 동일)
GRAPH_USAGE = os.getenv("GRAPH_USAGE", "anchored")


def _graph_to_query_expansion(graph_data: list, max_entities: int = 5) -> str:
    """Graph triple/path 에서 핵심 entity 추출 → query 확장 토큰.
    중복 제거, 등장 빈도 ↑ entity 우선."""
    if not graph_data:
        return ""
    is_path = isinstance(graph_data[0], dict) and "nodes" in graph_data[0]
    entity_count = defaultdict(int)
    if is_path:
        for p in graph_data:
            for n in (p.get("nodes") or []):
                if n and len(n) >= 2:
                    entity_count[n.strip()] += 1
    else:
        for t in graph_data:
            for k in ("from", "to"):
                v = (t.get(k) or "").strip()
                if v and len(v) >= 2:
                    entity_count[v] += 1
    top = sorted(entity_count.items(), key=lambda x: -x[1])[:max_entities]
    return " ".join(e for e, _ in top)


# ── Hybrid RAG 메인 ───────────────────────────────────────
def hybrid_rag(query, use_vector=True, use_graph=True, verbose=True):
    """Hybrid RAG 파이프라인.

    GRAPH_USAGE=expansion (기본, Phase 8 분석 결과):
      1. graph_search 로 entity 추출
      2. query + entity 로 vector 재검색 (확장 검색)
      3. LLM context = vector 본문만 (graph triple 미주입)
    GRAPH_USAGE=context (legacy):
      LLM prompt 에 graph triple 직접 포함 — 한국 행정 도메인엔 anti-pattern
    GRAPH_USAGE=off:
      graph 완전 무시 → vector-only 와 동일 (sanity check 용)
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[Hybrid RAG] 질문: {query}")
        print(f"  GRAPH_USAGE={GRAPH_USAGE}")
        print("=" * 60)

    vector_docs = []
    graph_relations = []
    graph_data = []

    # 1차: 원 query 로 vector + graph 검색
    if use_vector:
        try:
            vector_docs = vector_search(query, n_results=5)
        except Exception as e:
            if verbose:
                print(f"\n[Vector 검색 오류] {e}")
            vector_docs = []

    if use_graph and GRAPH_USAGE != "off":
        try:
            if GRAPH_USAGE == "anchored":
                # Phase 13: graph anchor 용 vector_docs 분리 옵션
                # USE_RERANK=0 (vector raw) + RERANK_FOR_GRAPH_ANCHOR=1 →
                # graph 시작점은 rerank 거친 chunks 사용 (topically cleaner)
                if (not USE_RERANK) and RERANK_FOR_GRAPH_ANCHOR and vector_docs:
                    try:
                        # 다시 top-10 가져와서 rerank → top-5 (graph anchor 전용)
                        anchor_candidates = _raw_vector_search(query, n_results=RETRIEVE_K)
                        anchor_docs = llm_rerank(query, anchor_candidates, top_k=RERANK_TO_K)
                    except Exception:
                        anchor_docs = vector_docs
                else:
                    anchor_docs = vector_docs
                # Phase 10b: max_relations 5→3 (graph 신호 더 압축)
                graph_relations = vector_anchored_graph_search(
                    query, anchor_docs, max_relations=3,
                )
                graph_data = graph_relations
            elif USE_GRAPH_SEARCH_V3:
                graph_data = graph_search_v3(query, vector_docs=vector_docs)
                graph_relations = paths_to_edges(graph_data)
            else:
                graph_relations = graph_search(query, vector_docs=vector_docs)
                graph_data = graph_relations
            if USE_GRAPH_GATING and vector_docs:
                graph_data = gate_graph_by_body(graph_data, vector_docs)
                graph_relations = (
                    paths_to_edges(graph_data) if USE_GRAPH_SEARCH_V3 else graph_data
                )
        except Exception as e:
            if verbose:
                print(f"\n[Graph 검색 오류] {e}")
            graph_data = []
            graph_relations = []

    # GRAPH_USAGE=expansion: graph entity 로 query 확장 후 vector 재검색
    expansion_used = False
    if GRAPH_USAGE == "expansion" and graph_data and use_vector:
        expansion = _graph_to_query_expansion(graph_data, max_entities=5)
        if expansion:
            try:
                expanded_q = f"{query}\n관련 키워드: {expansion}"
                expanded_docs = vector_search(expanded_q, n_results=5)
                # 원 검색 + 확장 검색 dedup 후 top-5 (확장 결과에 가중)
                seen = set()
                merged = []
                for d in expanded_docs + vector_docs:
                    key = (d.get("source", ""), (d.get("content", "") or "")[:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(d)
                    if len(merged) >= 5:
                        break
                vector_docs = merged
                expansion_used = True
                if verbose:
                    print(f"\n[Graph expansion] {expansion[:80]}")
            except Exception as e:
                if verbose:
                    print(f"\n[Graph expansion 오류] {e}")

    if verbose:
        print(f"\n[Vector 검색] {len(vector_docs)}개 문서")
        for doc in vector_docs:
            print(f"  - {doc.get('source','')} (유사도: {doc.get('score')})")
        print(f"\n[Graph] {len(graph_relations)}개 관계, expansion_used={expansion_used}")

    # GRAPH_USAGE=context|anchored 일 때 graph 를 LLM prompt 에 포함.
    # expansion 은 vector retrieval 만 보강 → LLM context 엔 vector 본문만.
    # off 면 graph 자체가 빈 상태.
    # USE_VERIFICATION=1 일 땐 vector-only context 로 draft 만든 후 graph 로 refine.
    if USE_VERIFICATION:
        # Phase 11: Chain-of-Verification — graph 는 검증 단계에만
        context = merge_results(vector_docs, [])  # draft 는 vector 만
        draft = generate_answer(query, context, model=GENERATION_MODEL)
        answer = verify_and_refine(query, draft, vector_docs, graph_relations)
    else:
        if GRAPH_USAGE in ("context", "anchored"):
            context = merge_results(vector_docs, graph_data)
        else:
            context = merge_results(vector_docs, [])
        answer = generate_answer(query, context, model=GENERATION_MODEL)

    if verbose:
        print(f"\n[최종 답변]\n{answer}")

    return {
        "query": query,
        "vector_docs": vector_docs,
        "graph_relations": graph_relations,
        "graph_paths": graph_data if USE_GRAPH_SEARCH_V3 else [],
        "context": context,
        "answer": answer,
        "graph_usage": GRAPH_USAGE,
        "expansion_used": expansion_used,
    }


def vector_only_rag(query, verbose=True):
    """Vector 단독 RAG"""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[Vector Only] 질문: {query}")
        print("=" * 60)

    vector_docs = vector_search(query, n_results=5)
    context = merge_results(vector_docs, [])
    answer = generate_answer(query, context, model=GENERATION_MODEL)

    if verbose:
        print(f"\n[Vector Only 답변]\n{answer}")

    return {
        "query": query,
        "vector_docs": vector_docs,
        "context": context,
        "answer": answer
    }

