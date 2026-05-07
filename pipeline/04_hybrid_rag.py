import os
import re
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_upstage import UpstageEmbeddings

# ── 경로 / 환경변수 ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# 실제 생성된 컬렉션 이름에 맞춤
EXPERIMENT_ID = "upstage_pro"  # 6개 실험 중 하나로 설정
COLLECTION_NAME = f"knu_cse_{EXPERIMENT_ID}"

upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")


def get_embedding_function():
    """문서 인덱싱용 (vector DB 빌드 시 사용)."""
    return UpstageEmbeddings(
        model="embedding-passage",  # 현행 표기 (이전 solar-embedding-1-large-passage 동일)
        api_key=UPSTAGE_API_KEY,
    )


def get_query_embedding_function():
    """쿼리 임베딩용 (vector_search 시 사용).
    Upstage 는 인덱싱(passage) 과 쿼리(query) 에 별도 모델을 권장.
    같은 임베딩 공간에 정렬돼 있어 retrieval 정밀도가 더 높음."""
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
def vector_search(query, n_results=3):
    """Chroma 에서 유사도 기반 검색.
    쿼리는 embedding-query, 인덱스는 embedding-passage (Upstage 권장 분리)."""
    embedding_fn = get_query_embedding_function()
    client = get_chroma_client()

    # Chroma v0.6 부터 list_collections() 가 collection name 문자열만 반환.
    # 이전 코드의 c.name 접근은 NotImplementedError 를 raise.
    existing_collections = list(client.list_collections())
    if COLLECTION_NAME not in existing_collections:
        raise ValueError(
            f"컬렉션이 없습니다: {COLLECTION_NAME} | 현재 컬렉션: {existing_collections}"
        )

    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedding_fn.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
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
            "score": score
        })

    return docs


# ── 질문 키워드 추출 ─────────────────────────────────────
def extract_keywords(query):
    """질문에서 그래프 검색용 핵심 키워드 추출.
    조건해석형 질의 (e.g. "X 이수하면 Y 가능한가") 의 조건절 단어
    ('하면', '되는가요' 등) 는 graph 노드 매칭에 도움 안 되어 제외하지만,
    'X학점' / '3회' 같은 숫자+단위 짧은 토큰은 살린다 (조건 판정 핵심)."""
    stopwords = {
        "어떻게", "무엇", "뭐", "인가요", "있나요", "해주세요", "알려줘",
        "알려주세요", "가능한가요", "되는가요", "관련", "대한", "에서",
        "으로", "를", "을", "이", "가", "은", "는", "와", "과", "좀",
        "무슨", "어떤", "정도", "대한", "하고", "하면", "하나요"
    }

    cleaned = re.sub(r"[^\w\s]", " ", query)
    tokens = []

    for token in cleaned.split():
        token = token.strip()
        # 2자 미만 컷, 단 숫자 포함 토큰 (학점/회차 등) 은 1자도 살림
        if len(token) < 2 and not re.search(r"\d", token):
            continue
        if token in stopwords:
            continue
        tokens.append(token)

    # 중복 제거하면서 순서 유지
    unique_tokens = []
    seen = set()
    for token in tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    # 긴 키워드(고유명사/복합어일 가능성)를 우선해서 graph 매칭 정밀도 향상.
    # "스타트업" 같은 짧은 일반어보다 "K-스타트업 2025" 같은 specific 토큰을 먼저 사용.
    # top-k 5 → 8 로 확대 (한국어는 띄어쓰기 단위로 7-12 토큰이 흔함, 5 컷이면
    # 조건해석형 질의의 핵심 어휘가 잘림).
    unique_tokens.sort(key=lambda t: -len(t))
    return unique_tokens[:8]


# ── Graph 검색 ───────────────────────────────────────────
def _score_relation(rel, keywords):
    """질문 키워드와의 매칭 점수. 정확 일치 + 부분 일치 길이 비율 가중.
    이전엔 정확 +3 / 부분 +1 단순 점수라, LLM 이 만든 긴 노드명
    ('2026년도 컴퓨터학부 조기졸업요건' 등) 에서 정확 일치가 거의 안 나와
    대부분 +1 로 깔려 정렬이 무의미했음. 부분 일치는 키워드가 노드명에서
    차지하는 비중에 비례해 가중 (specific 매칭일수록 점수 ↑)."""
    score = 0.0
    from_node = rel.get("from", "")
    to_node = rel.get("to", "")
    for kw in keywords:
        if not kw:
            continue
        # 양쪽 노드 정확 일치 = 강한 단서
        if kw == from_node:
            score += 5
        elif kw in from_node:
            score += 1 + (len(kw) / max(len(from_node), 1)) * 2
        if kw == to_node:
            score += 5
        elif kw in to_node:
            score += 1 + (len(kw) / max(len(to_node), 1)) * 2
    return score


# 의미 관계 화이트리스트 — MENTIONS 같은 메타 관계 제외하고 실제 의미를
# 가진 관계만 graph traversal 에 포함. (MENTIONS 는 전체 관계의 55.8% 차지하나
# Document → Entity 자동 메타라 hybrid 답변에 노이즈로 작용.)
MEANINGFUL_RELATIONS = [
    "REQUIRES", "HAS_CONDITION", "HAS_DEADLINE", "TARGETS",
    "OFFERS", "PROVIDES", "INCLUDES", "PART_OF", "BELONGS_TO",
    "APPLIES_TO", "EXCLUDES", "RELATED_TO", "CHARGES",
    "REWARDS", "ACCEPTS", "REFERS",
]

# 조건/대상 판정에 결정적인 관계 타입에 가중치 부여
RELATION_TYPE_WEIGHT = {
    "REQUIRES": 1.5, "HAS_CONDITION": 1.5,
    "TARGETS": 1.3, "HAS_DEADLINE": 1.3,
    "OFFERS": 1.2, "PROVIDES": 1.2, "INCLUDES": 1.2,
    "EXCLUDES": 1.2, "APPLIES_TO": 1.2,
    "PART_OF": 1.0, "BELONGS_TO": 1.0,
    "RELATED_TO": 0.8, "CHARGES": 1.1,
    "REWARDS": 1.0, "ACCEPTS": 1.0, "REFERS": 0.8,
}


def graph_search(query, vector_docs=None, max_relations=5):
    """Vector-anchored, relation-centric GraphRAG.

    이전 keyword CONTAINS 기반 → 동의어/약어 못 잡고, MENTIONS 메타관계가
    노이즈로 들어가던 문제 해결.

    설계:
    1. Anchor: vector_docs 의 출처 문서가 MENTIONS 한 entity 들이 graph 진입점
       → vector 가 이미 "관련 있다" 인정한 entity 만 사용 (정밀도 ↑)
    2. Relation-centric expansion: anchor 에서 1-hop, 단 의미 관계만
       (MEANINGFUL_RELATIONS), Document 노드 제외
    3. Score = anchor confidence × relation type weight × subgraph cohesion
       × hub penalty (일반 entity 디스카운트)

    vector_docs 가 None 이면 keyword 기반 fallback (호환성).
    """
    if not vector_docs:
        return []  # vector 없으면 anchor 없음 — graph 검색 불가

    driver = get_neo4j_driver()
    triples = []
    seen = {}

    try:
        with driver.session() as session:
            # [1] Anchor: vector 가 가져온 문서들이 mention 한 entity
            files = list({d.get("source", "") for d in vector_docs if d.get("source")})
            if not files:
                return []

            anchor_rows = session.run("""
                MATCH (d:Document)-[:MENTIONS]->(e)
                WHERE d.file_name IN $files AND NOT e:Document
                RETURN DISTINCT
                    coalesce(e.name, '') AS name,
                    elementId(e) AS id
            """, {"files": files}).data()

            anchors = [(r["name"], r["id"]) for r in anchor_rows if r.get("name")]
            if not anchors:
                return []
            anchor_set = {n for n, _ in anchors}

            # [2] Anchor 별 1-hop 의미 관계 expansion (양방향, MENTIONS 제외)
            for anchor_name, anchor_id in anchors:
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
                """, {"id": anchor_id, "types": MEANINGFUL_RELATIONS}).data()

                for row in rows:
                    from_n = (row.get("from_name") or "").strip()
                    to_n = (row.get("to_name") or "").strip()
                    rel = (row.get("rel") or "").strip()
                    neighbor = (row.get("neighbor_name") or "").strip()
                    neighbor_deg = row.get("neighbor_degree") or 1

                    if not from_n or not to_n or not rel:
                        continue

                    # 점수 계산
                    score = 1.0  # base from anchor

                    # (a) Relation type 가중
                    score *= RELATION_TYPE_WEIGHT.get(rel, 1.0)

                    # (b) Subgraph cohesion: neighbor 도 anchor 면 강한 신호
                    if neighbor in anchor_set:
                        score *= 1.5

                    # (c) Hub penalty: '학생', '경북대학교' 같은 일반 entity
                    if neighbor_deg > 20:
                        score *= 0.5
                    elif neighbor_deg > 10:
                        score *= 0.7

                    key = (from_n, rel, to_n)
                    # 중복 시 더 높은 점수만 유지
                    if key not in seen or seen[key]["score"] < score:
                        seen[key] = {
                            "from": from_n, "relation": rel, "to": to_n,
                            "score": score,
                        }
    finally:
        driver.close()

    # 점수순 정렬, top-k. 동점은 from 명 알파벳 순 (재현성).
    triples = sorted(seen.values(), key=lambda x: (-x["score"], x["from"]))
    # score 필드는 외부 contract 와 다르므로 제거
    result = []
    for t in triples[:max_relations]:
        result.append({"from": t["from"], "relation": t["relation"], "to": t["to"]})
    return result


# ── Graph 관계 타입을 한국어 술어로 변환 (LLM 가독성 ↑) ──────
# "X --REQUIRES--> Y" 같은 ASCII art 보다 자연어가 답변 활용도 높음.
REL_KO = {
    "REQUIRES":      "는 다음을 요구함:",
    "HAS_DEADLINE":  "의 마감일:",
    "HAS_CONDITION": "의 조건:",
    "TARGETS":       "의 대상:",
    "PROVIDES":      "는 다음을 제공함:",
    "OFFERS":        "는 다음을 제공:",
    "REWARDS":       "는 다음에 보상함:",
    "PART_OF":       "는 다음의 일부:",
    "BELONGS_TO":    "는 다음에 속함:",
    "APPLIES_TO":    "는 다음에 적용됨:",
    "RELATED_TO":    "는 관련됨:",
    "MENTIONS":      "에서 언급됨:",
}


# ── 결과 통합 ─────────────────────────────────────────────
def merge_results(vector_docs, graph_relations):
    """Vector + Graph 결과를 컨텍스트 문자열로 통합.

    핵심 변경: graph 블록을 vector 보다 *앞*에 배치 (위치 가중치로 인한
    graph 신호 약화 방지). 조건/관계 판정에서는 graph 가 결정적이며,
    서술/배경 정보는 vector 본문에서 가져오는 게 자연스럽다.
    triple 표현은 한국어 술어로 변환해 LLM 가독성/활용도 향상.
    """
    context_parts = []

    if graph_relations:
        graph_text = "=== 핵심 관계/조건 단서 (조건·자격·대상 판정용) ===\n"
        for rel in graph_relations:
            if rel["from"] and rel["to"]:
                verb = REL_KO.get(rel["relation"], f"--[{rel['relation']}]-->")
                graph_text += f"- {rel['from']} {verb} {rel['to']}\n"
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
GENERATION_MAX_TOKENS = None     # None=모델 기본값 (solar-pro3 ~ 4k), 정수면 cap


def generate_answer(query, context, model=GENERATION_MODEL):
    """컨텍스트 기반 LLM 답변 생성. 모델/온도/max_tokens 는 모듈 상수 참조."""
    if not context.strip():
        return "해당 정보를 찾을 수 없습니다."
    prompt = f"""당신은 경북대학교 컴퓨터학부 학생들을 위한 AI 챗봇입니다.

아래 검색 결과를 바탕으로 질문에 답변하세요. 검색 결과는 두 가지로 구성됩니다:
- [핵심 관계/조건 단서]: 지식 그래프에서 추출한 개체 간 조건·자격·대상·요구사항 관계. **조건 판정형 질문(자격/요건/대상/기한 등)에서 결정적 근거**로 사용하세요. 본문에 명시 안 된 조건이나 관계도 단서가 일치하면 답변에 반영하세요.
- [문서 본문]: 공지/문서의 실제 서술. **배경 설명·구체 수치·인용**의 근거로 사용하세요.

두 정보가 충돌할 경우: 사실(날짜·금액·수치)은 본문이 우선, 자격/대상/조건 관계는 단서가 우선.
반드시 검색 결과에 근거해서만 답변하세요. 검색 결과에 없는 내용은 추측하지 말고 "해당 정보를 찾을 수 없습니다."라고 답변하세요.

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


# ── Hybrid RAG 메인 ───────────────────────────────────────
def hybrid_rag(query, use_vector=True, use_graph=True, verbose=True):
    """Hybrid RAG 파이프라인 실행"""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[Hybrid RAG] 질문: {query}")
        print("=" * 60)

    vector_docs = []
    graph_relations = []

    if use_vector:
        try:
            vector_docs = vector_search(query, n_results=3)
            if verbose:
                print(f"\n[Vector 검색] {len(vector_docs)}개 문서 검색됨")
                for doc in vector_docs:
                    print(f"  - {doc['source']} (유사도: {doc['score']})")
        except Exception as e:
            if verbose:
                print(f"\n[Vector 검색 오류] {e}")
            vector_docs = []

    if use_graph:
        try:
            # vector 결과를 anchor 로 활용 (vector-anchored GraphRAG)
            graph_relations = graph_search(query, vector_docs=vector_docs)
            if verbose:
                print(f"\n[Graph 검색] {len(graph_relations)}개 관계 탐색됨")
                for rel in graph_relations[:5]:
                    print(f"  - {rel['from']} --[{rel['relation']}]--> {rel['to']}")
        except Exception as e:
            if verbose:
                print(f"\n[Graph 검색 오류] {e}")
            graph_relations = []

    context = merge_results(vector_docs, graph_relations)
    answer = generate_answer(query, context, model=GENERATION_MODEL)

    if verbose:
        print(f"\n[최종 답변]\n{answer}")

    return {
        "query": query,
        "vector_docs": vector_docs,
        "graph_relations": graph_relations,
        "context": context,
        "answer": answer
    }


def vector_only_rag(query, verbose=True):
    """Vector 단독 RAG"""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[Vector Only] 질문: {query}")
        print("=" * 60)

    vector_docs = vector_search(query, n_results=3)
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

