import os
import re
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# ── 경로 / 환경변수 ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 실제 생성된 컬렉션 이름에 맞춤
EXPERIMENT_ID = "openai_small"
COLLECTION_NAME = f"knu_cse_{EXPERIMENT_ID}"

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding_function():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
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
    """Chroma에서 유사도 기반 검색"""
    embedding_fn = get_embedding_function()
    client = get_chroma_client()

    existing_collections = client.list_collections()
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
    """질문에서 그래프 검색용 핵심 키워드 추출"""
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
        if len(token) < 2:
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

    return unique_tokens[:5]


# ── Graph 검색 ───────────────────────────────────────────
def graph_search(query, max_nodes_per_keyword=5, max_relations=20):
    """Neo4j에서 키워드 기반 관계 탐색"""
    driver = get_neo4j_driver()
    results = []
    seen = set()
    keywords = extract_keywords(query)

    with driver.session() as session:
        for keyword in keywords:
            node_result = session.run("""
                MATCH (n)
                WHERE coalesce(n.name, '') CONTAINS $keyword
                   OR coalesce(n.title, '') CONTAINS $keyword
                   OR coalesce(n.file_name, '') CONTAINS $keyword
                RETURN
                    coalesce(n.name, n.title, n.file_name, '') AS node_name,
                    labels(n) AS labels
                LIMIT $limit
            """, {
                "keyword": keyword,
                "limit": max_nodes_per_keyword
            })

            nodes = node_result.data()

            for node in nodes:
                node_name = (node.get("node_name") or "").strip()
                if not node_name:
                    continue

                # 정방향 관계
                rel_result = session.run("""
                    MATCH (a)-[r]->(b)
                    WHERE coalesce(a.name, a.title, a.file_name, '') = $node_name
                    RETURN
                        coalesce(a.name, a.title, a.file_name, '') AS from_node,
                        type(r) AS rel,
                        coalesce(b.name, b.title, b.file_name, '') AS to_node
                    LIMIT 10
                """, {"node_name": node_name})

                for row in rel_result.data():
                    from_node = (row.get("from_node") or "").strip()
                    rel = (row.get("rel") or "").strip()
                    to_node = (row.get("to_node") or "").strip()

                    if not from_node or not to_node:
                        continue

                    key = (from_node, rel, to_node)
                    if key not in seen:
                        results.append({
                            "from": from_node,
                            "relation": rel,
                            "to": to_node
                        })
                        seen.add(key)

                # 역방향 관계
                rev_result = session.run("""
                    MATCH (a)-[r]->(b)
                    WHERE coalesce(b.name, b.title, b.file_name, '') = $node_name
                    RETURN
                        coalesce(a.name, a.title, a.file_name, '') AS from_node,
                        type(r) AS rel,
                        coalesce(b.name, b.title, b.file_name, '') AS to_node
                    LIMIT 10
                """, {"node_name": node_name})

                for row in rev_result.data():
                    from_node = (row.get("from_node") or "").strip()
                    rel = (row.get("rel") or "").strip()
                    to_node = (row.get("to_node") or "").strip()

                    if not from_node or not to_node:
                        continue

                    key = (from_node, rel, to_node)
                    if key not in seen:
                        results.append({
                            "from": from_node,
                            "relation": rel,
                            "to": to_node
                        })
                        seen.add(key)

    driver.close()
    return results[:max_relations]


# ── 결과 통합 ─────────────────────────────────────────────
def merge_results(vector_docs, graph_relations):
    """Vector + Graph 결과를 컨텍스트 문자열로 통합"""
    context_parts = []

    if vector_docs:
        vector_text = "=== 문서 검색 결과 ===\n"
        for i, doc in enumerate(vector_docs, start=1):
            vector_text += f"\n[{i}] 출처: {doc['source']} (유사도: {doc['score']})\n"
            vector_text += f"{doc['content']}\n"
        context_parts.append(vector_text.strip())

    if graph_relations:
        graph_text = "=== 관계 그래프 검색 결과 ===\n"
        for rel in graph_relations:
            if rel["from"] and rel["to"]:
                graph_text += f"- {rel['from']} --[{rel['relation']}]--> {rel['to']}\n"
        context_parts.append(graph_text.strip())

    return "\n\n".join(context_parts).strip()


# ── LLM 답변 생성 ─────────────────────────────────────────
def generate_answer(query, context, model="gpt-4o-mini"):
    """컨텍스트 기반 LLM 답변 생성"""
    prompt = f"""당신은 경북대학교 컴퓨터학부 학생들을 위한 AI 챗봇입니다.
아래 검색된 문서 내용과 그래프 관계 정보를 바탕으로 질문에 답변하세요.
반드시 검색 결과에 근거해서만 답변하세요.
검색 결과에 없는 내용은 추측하지 말고 "해당 정보를 찾을 수 없습니다."라고 답변하세요.

[검색 결과]
{context}

[질문]
{query}

[답변]
"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

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
        vector_docs = vector_search(query, n_results=3)
        if verbose:
            print(f"\n[Vector 검색] {len(vector_docs)}개 문서 검색됨")
            for doc in vector_docs:
                print(f"  - {doc['source']} (유사도: {doc['score']})")

    if use_graph:
        graph_relations = graph_search(query)
        if verbose:
            print(f"\n[Graph 검색] {len(graph_relations)}개 관계 탐색됨")
            for rel in graph_relations[:5]:
                print(f"  - {rel['from']} --[{rel['relation']}]--> {rel['to']}")

    context = merge_results(vector_docs, graph_relations)
    answer = generate_answer(query, context)

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
    answer = generate_answer(query, context)

    if verbose:
        print(f"\n[Vector Only 답변]\n{answer}")

    return {
        "query": query,
        "vector_docs": vector_docs,
        "context": context,
        "answer": answer
    }

