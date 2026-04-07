import os
import json
from dotenv import load_dotenv
import chromadb
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_PROVIDER = "openai"


def get_embedding_function():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )


def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Vector 검색 ──────────────────────────────────────────
def vector_search(query, n_results=3):
    """Chroma에서 유사도 기반 검색"""
    embedding_fn = get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(f"knu_cse_{EMBEDDING_PROVIDER}")

    query_embedding = embedding_fn.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "content": doc,
            "source": meta.get("file_name", ""),
            "score": round(1 - dist, 3)
        })

    return docs


# ── Graph 검색 ───────────────────────────────────────────
def graph_search(query):
    """Neo4j에서 키워드 기반 관계 탐색"""
    driver = get_neo4j_driver()
    results = []

    with driver.session() as session:
        # 1. 키워드로 관련 노드 찾기
        node_result = session.run("""
            MATCH (n)
            WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $query)
            RETURN n.name AS name, labels(n)[0] AS type
            LIMIT 5
        """, {"query": query})

        nodes = node_result.data()

        # 2. 찾은 노드의 관계 탐색 (멀티홉)
        for node in nodes:
            if not node.get("name"):
                continue

            rel_result = session.run("""
                MATCH (a {name: $name})-[r]->(b)
                RETURN a.name AS from, type(r) AS rel, b.name AS to
                LIMIT 5
            """, {"name": node["name"]})

            for row in rel_result.data():
                results.append({
                    "from": row["from"],
                    "relation": row["rel"],
                    "to": row["to"]
                })

            # 역방향 관계도 탐색
            rev_result = session.run("""
                MATCH (a)-[r]->(b {name: $name})
                RETURN a.name AS from, type(r) AS rel, b.name AS to
                LIMIT 5
            """, {"name": node["name"]})

            for row in rev_result.data():
                results.append({
                    "from": row["from"],
                    "relation": row["rel"],
                    "to": row["to"]
                })

    driver.close()
    return results


# ── 결과 통합 ─────────────────────────────────────────────
def merge_results(vector_docs, graph_relations):
    """Vector + Graph 결과를 컨텍스트 문자열로 통합"""
    context = ""

    if vector_docs:
        context += "=== 문서 검색 결과 ===\n"
        for i, doc in enumerate(vector_docs):
            context += f"\n[{i+1}] 출처: {doc['source']} (유사도: {doc['score']})\n"
            context += f"{doc['content']}\n"

    if graph_relations:
        context += "\n=== 관계 그래프 검색 결과 ===\n"
        seen = set()
        for rel in graph_relations:
            key = f"{rel['from']}-{rel['relation']}-{rel['to']}"
            if key not in seen and rel['from'] and rel['to']:
                context += f"- {rel['from']} --[{rel['relation']}]--> {rel['to']}\n"
                seen.add(key)

    return context


# ── LLM 답변 생성 ─────────────────────────────────────────
def generate_answer(query, context, model="gpt-4o-mini"):
    """컨텍스트 기반 LLM 답변 생성"""
    prompt = f"""당신은 경북대학교 컴퓨터학부 학생들을 위한 AI 챗봇입니다.
아래 검색된 문서와 관계 정보를 바탕으로 질문에 정확하게 답변해주세요.
검색 결과에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답변하세요.

[검색 결과]
{context}

[질문]
{query}

[답변]"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# ── Hybrid RAG 메인 ───────────────────────────────────────
def hybrid_rag(query, use_vector=True, use_graph=True, verbose=True):
    """Hybrid RAG 파이프라인 실행"""
    if verbose:
        print(f"\n{'='*50}")
        print(f"질문: {query}")
        print('='*50)

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
        "answer": answer
    }


def vector_only_rag(query, verbose=True):
    """Vector 단독 RAG (비교 실험용)"""
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Vector Only] 질문: {query}")
        print('='*50)

    vector_docs = vector_search(query, n_results=3)
    context = merge_results(vector_docs, [])
    answer = generate_answer(query, context)

    if verbose:
        print(f"\n[Vector Only 답변]\n{answer}")

    return {
        "query": query,
        "vector_docs": vector_docs,
        "answer": answer
    }


if __name__ == "__main__":
    test_queries = [
        "글솝 졸업요건에서 기술창업역량을 어떻게 충족할 수 있나요?",
        "교양초과이수 교과구분 변경 신청은 어떻게 하나요?",
        "창업 관련 교과목에는 어떤 것들이 있나요?"
    ]

    for query in test_queries:
        print("\n" + "="*60)
        print("[ Hybrid RAG ]")
        hybrid_rag(query)

        print("\n[ Vector Only RAG ]")
        vector_only_rag(query)
        print("\n")