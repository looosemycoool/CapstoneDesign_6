import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI
from vector_rag import load_documents, split_documents, build_vectorstore


# =========================
# 0. 환경변수 로드
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")


# =========================
# 1. 기본 설정
# =========================
DATA_DIR = Path("./data")
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"


# =========================
# 2. Neo4j 연결
# =========================
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "test1234")
)


# =========================
# 3. 공용 LLM
# =========================
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)


# =========================
# 4. JSON 파싱 유틸
# =========================
def parse_json_safely(text: str, default):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # JSON 객체/배열 부분만 최대한 추출
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)

    candidate = text
    if obj_match:
        candidate = obj_match.group(0)
    elif arr_match:
        candidate = arr_match.group(0)

    try:
        return json.loads(candidate)
    except Exception:
        return default


# =========================
# 5. 질문 라우팅
# =========================
def route_question(question: str) -> dict:
    """
    질문을 보고 VECTOR / GRAPH / HYBRID 중 하나를 선택
    """
    prompt = f"""
너는 RAG retrieval router다.

질문을 보고 아래 셋 중 하나를 골라라.

- VECTOR:
  문서 원문에서 사실/수치/정의/공지 내용을 찾는 것이 더 적합한 질문
  예: 몇 학점인가, 언제 가능한가, 무엇인가

- GRAPH:
  조건/비교/관계/구성요소/연결 구조를 파악하는 것이 더 적합한 질문
  예: 차이는 무엇인가, 어떤 요소로 구성되는가, 어떤 관계인가

- HYBRID:
  원문 근거도 필요하고 관계 구조도 같이 보는 것이 더 좋은 질문

반드시 JSON만 출력:
{{
  "route": "VECTOR 또는 GRAPH 또는 HYBRID",
  "reason": "짧은 이유"
}}

질문:
{question}
"""
    response = llm.invoke(prompt).content
    data = parse_json_safely(response, {"route": "HYBRID", "reason": "default"})

    route = str(data.get("route", "HYBRID")).upper().strip()
    if route not in {"VECTOR", "GRAPH", "HYBRID"}:
        route = "HYBRID"

    return {
        "route": route,
        "reason": str(data.get("reason", ""))
    }


# =========================
# 6. 질문에서 그래프 엔티티 추출
# =========================
def extract_graph_entities(question: str) -> list[str]:
    """
    질문에서 Neo4j 그래프에서 찾을 만한 핵심 엔티티명을 추출
    """
    prompt = f"""
다음 질문에서 그래프 DB 조회에 사용할 핵심 엔티티명만 추출하라.

규칙:
1. 한국어 원문 표현을 최대한 유지
2. 너무 일반적인 단어는 제외
3. 최대 5개까지
4. 없으면 빈 배열
5. 설명 없이 JSON 배열만 출력

예:
["졸업요건", "기술창업역량"]

질문:
{question}
"""
    response = llm.invoke(prompt).content
    entities = parse_json_safely(response, [])

    if not isinstance(entities, list):
        return []

    cleaned = []
    for e in entities:
        e = str(e).strip()
        if not e:
            continue
        if e not in cleaned:
            cleaned.append(e)

    return cleaned[:5]


# =========================
# 7. Neo4j 쿼리 실행
# =========================
def run_graph_query(query: str, params: dict | None = None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


# =========================
# 8. 벡터 문맥 검색
# =========================
def retrieve_vector_context(vectorstore, question: str, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([
        f"[벡터 출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    return context, retrieved_docs


# =========================
# 9. 그래프 문맥 검색
# =========================
def retrieve_graph_context(question: str) -> dict:
    entities = extract_graph_entities(question)

    if not entities:
        return {
            "entities": [],
            "context": "그래프 검색 결과 없음"
        }

    seen = set()
    lines = []

    for entity in entities:
        # 1) 정확히 같은 이름 노드
        exact_query = """
        MATCH (a {name: $name})-[r]->(b)
        RETURN a.name AS source, labels(a) AS source_labels,
               type(r) AS rel,
               b.name AS target, labels(b) AS target_labels
        LIMIT 30
        """
        exact_rows = run_graph_query(exact_query, {"name": entity})

        for row in exact_rows:
            line = (
                f"{row['source']} ({', '.join(row['source_labels'])}) "
                f"-[{row['rel']}]-> "
                f"{row['target']} ({', '.join(row['target_labels'])})"
            )
            if line not in seen:
                seen.add(line)
                lines.append(line)

        # 2) 이름 포함 검색
        contain_query = """
        MATCH (a)-[r]->(b)
        WHERE a.name CONTAINS $kw OR b.name CONTAINS $kw
        RETURN a.name AS source, labels(a) AS source_labels,
               type(r) AS rel,
               b.name AS target, labels(b) AS target_labels
        LIMIT 30
        """
        contain_rows = run_graph_query(contain_query, {"kw": entity})

        for row in contain_rows:
            line = (
                f"{row['source']} ({', '.join(row['source_labels'])}) "
                f"-[{row['rel']}]-> "
                f"{row['target']} ({', '.join(row['target_labels'])})"
            )
            if line not in seen:
                seen.add(line)
                lines.append(line)

    # 비교형 질문이면 DIFFERS_FROM 보조 조회
    if any(word in question for word in ["차이", "비교", "다른", "어떻게 다른가"]):
        diff_query = """
        MATCH (a)-[r:DIFFERS_FROM]->(b)
        RETURN a.name AS source, type(r) AS rel, b.name AS target
        LIMIT 20
        """
        diff_rows = run_graph_query(diff_query)

        for row in diff_rows:
            line = f"{row['source']} -[{row['rel']}]-> {row['target']}"
            if line not in seen:
                seen.add(line)
                lines.append(line)

    if not lines:
        graph_context = "그래프 검색 결과 없음"
    else:
        graph_context = "[그래프 검색 결과]\n" + "\n".join(lines)

    return {
        "entities": entities,
        "context": graph_context
    }


# =========================
# 10. 벡터 출처 정리
# =========================
def summarize_vector_sources(retrieved_docs):
    sources = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)
    return sources


# =========================
# 11. 최종 답변 생성
# =========================
def generate_answer(
    question: str,
    route_info: dict,
    vector_context: str,
    graph_context: str,
    vector_sources: list[str],
    graph_entities: list[str]
):
    prompt = f"""
너는 경북대학교 컴퓨터학부 학사제도 문서를 기반으로 답하는 하이브리드 RAG 도우미다.

이번 질문에 대해 retrieval router가 선택한 전략은 다음과 같다.
- route: {route_info["route"]}
- reason: {route_info["reason"]}

너는 아래 규칙을 반드시 지켜라.

[답변 규칙]
1. route가 VECTOR면 벡터 검색 결과를 우선 신뢰하라.
2. route가 GRAPH면 그래프 검색 결과를 우선 신뢰하라.
3. route가 HYBRID면 두 정보를 함께 참고하라.
4. 문서/그래프 근거가 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다."라고 답하라.
5. 숫자, 학점, 평점, 조건은 정확히 써라.
6. 불필요한 장황한 설명은 하지 마라.
7. 답변 마지막에 반드시 아래 형식으로 붙여라.
8. 그래프 검색 결과에서 상위 조건과 하위 조건이 함께 보이면, 하위 조건을 무조건 졸업요건의 직접 필수조건으로 단정하지 말고 계층적으로 해석하라.
9. 질문이 "졸업요건 전체"를 묻는 경우에는 가장 상위 조건만 우선 정리하고, 세부 대체항목은 별도로 구분해서 설명하라.

[참고]
- 벡터: 파일명들
- 그래프 엔티티: 엔티티들

[질문]
{question}

[벡터 검색 결과]
{vector_context}

[그래프 검색 결과]
{graph_context}
"""
    response = llm.invoke(prompt)
    return response.content


# =========================
# 12. 하이브리드 질의
# =========================
def ask_hybrid(vectorstore, question: str):
    route_info = route_question(question)

    vector_context = ""
    retrieved_docs = []
    graph_result = {"entities": [], "context": "그래프 검색 결과 없음"}

    if route_info["route"] == "VECTOR":
        vector_context, retrieved_docs = retrieve_vector_context(vectorstore, question, k=4)

    elif route_info["route"] == "GRAPH":
        graph_result = retrieve_graph_context(question)

    else:  # HYBRID
        vector_context, retrieved_docs = retrieve_vector_context(vectorstore, question, k=4)
        graph_result = retrieve_graph_context(question)

    vector_sources = summarize_vector_sources(retrieved_docs)

    answer = generate_answer(
        question=question,
        route_info=route_info,
        vector_context=vector_context if vector_context else "벡터 검색 결과 없음",
        graph_context=graph_result["context"],
        vector_sources=vector_sources,
        graph_entities=graph_result["entities"]
    )

    return {
        "route": route_info["route"],
        "route_reason": route_info["reason"],
        "graph_entities": graph_result["entities"],
        "graph_context": graph_result["context"],
        "vector_sources": vector_sources,
        "answer": answer
    }


# =========================
# 13. 메인
# =========================
def main():
    print("1) 문서 로드 중...")
    documents = load_documents(DATA_DIR)
    print(f"로드된 문서 수: {len(documents)}")

    print("2) 문서 분할 중...")
    split_docs = split_documents(documents)
    print(f"청크 수: {len(split_docs)}")

    print("3) 벡터DB 생성 중...")
    vectorstore = build_vectorstore(split_docs, EMBEDDING_MODEL)
    print("벡터DB 생성 완료")

    print("4) 하이브리드 질의 시작")
    print("종료하려면 exit 입력")

    while True:
        question = input("\n질문: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("종료합니다.")
            break

        result = ask_hybrid(vectorstore, question)

        print("\n" + "=" * 80)
        print("[라우팅 결과]")
        print("route:", result["route"])
        print("reason:", result["route_reason"])

        print("\n[그래프 엔티티 추출]")
        print(result["graph_entities"])

        print("\n[벡터 참고 문서]")
        if result["vector_sources"]:
            for src in result["vector_sources"]:
                print("-", src)
        else:
            print("없음")

        print("\n[그래프 문맥]")
        print(result["graph_context"])

        print("\n[최종 답변]")
        print(result["answer"])
        print("=" * 80)


if __name__ == "__main__":
    main()