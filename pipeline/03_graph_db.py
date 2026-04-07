import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_db(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[초기화] 기존 그래프 삭제 완료")


def extract_entities_and_relations(file_name, text):
    """LLM으로 파일에서 개체·관계 추출"""
    prompt = f"""다음 경북대학교 컴퓨터학부 문서에서 개체와 관계를 추출해주세요.

파일명: {file_name}
내용:
{text[:2000]}

다음 JSON 형식으로만 응답해주세요 (다른 텍스트 없이):
{{
  "entities": [
    {{"name": "개체명", "type": "개체타입", "properties": {{"key": "value"}}}}
  ],
  "relations": [
    {{"from": "개체명1", "to": "개체명2", "type": "관계타입", "properties": {{"key": "value"}}}}
  ]
}}

개체 타입: Document(문서), Scholarship(장학금), Schedule(일정), Requirement(요건), Department(학과), Program(프로그램), Course(교과목), Student(학생대상)
관계 타입: HAS_DEADLINE(마감일), REQUIRES(요구), RELATED_TO(관련), PART_OF(소속), APPLIES_TO(적용대상), HAS_CONDITION(조건)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        if "```" in result:
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        return json.loads(result)
    except Exception as e:
        print(f"  [LLM 오류] {e}")
        return {"entities": [], "relations": []}


def create_document_node(session, file_name):
    """문서 노드 생성"""
    session.run("""
        MERGE (d:Document {file_name: $file_name})
        SET d.file_name = $file_name
    """, {"file_name": file_name})


def create_entities_and_relations(session, file_name, extracted):
    """추출된 개체·관계를 Neo4j에 저장"""
    # 개체 생성
    for entity in extracted.get("entities", []):
        name = entity.get("name", "").strip()
        etype = entity.get("type", "Entity")
        props = entity.get("properties", {})
        if not name:
            continue

        session.run(f"""
            MERGE (e:{etype} {{name: $name}})
        """, {"name": name})

        # 문서 → 개체 연결
        session.run("""
            MATCH (d:Document {file_name: $file_name})
            MATCH (e {name: $name})
            MERGE (d)-[:MENTIONS]->(e)
        """, {"file_name": file_name, "name": name})

    # 관계 생성
    for rel in extracted.get("relations", []):
        from_name = rel.get("from", "").strip()
        to_name = rel.get("to", "").strip()
        rel_type = rel.get("type", "RELATED_TO").strip().replace(" ", "_")
        if not from_name or not to_name:
            continue

        session.run(f"""
            MATCH (a {{name: $from_name}})
            MATCH (b {{name: $to_name}})
            MERGE (a)-[:{rel_type}]->(b)
        """, {"from_name": from_name, "to_name": to_name})


def build_graph():
    """전체 그래프 구축 - manual_parsed.json 사용"""
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")
    with open(manual_path, encoding="utf-8") as f:
        manual_files = json.load(f)

    driver = get_driver()
    clear_db(driver)

    print(f"[시작] Graph DB 구축 ({len(manual_files)}개 파일)")

    with driver.session() as session:
        for i, mf in enumerate(manual_files):
            file_name = mf["file_name"]
            parsed_text = mf.get("parsed_text", "").strip()

            if not parsed_text:
                print(f"[{i+1}/{len(manual_files)}] {file_name[:40]} → 텍스트 없음 스킵")
                continue

            print(f"\n[{i+1}/{len(manual_files)}] {file_name[:40]}")

            # 1. 문서 노드 생성
            create_document_node(session, file_name)

            # 2. LLM으로 개체·관계 추출
            extracted = extract_entities_and_relations(file_name, parsed_text)
            entities_count = len(extracted.get("entities", []))
            relations_count = len(extracted.get("relations", []))
            print(f"  개체: {entities_count}개, 관계: {relations_count}개 추출")

            # 3. Neo4j에 저장
            create_entities_and_relations(session, file_name, extracted)

    # 통계
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

    print(f"\n[완료] Graph DB 구축 완료!")
    print(f"총 노드: {node_count}개")
    print(f"총 관계: {rel_count}개")

    driver.close()


def search_graph(keyword):
    """그래프 검색 테스트"""
    driver = get_driver()
    print(f"\n[그래프 검색] '{keyword}'")

    with driver.session() as session:
        result = session.run("""
            MATCH (n)-[r]->(m)
            WHERE n.name CONTAINS $keyword
               OR n.title CONTAINS $keyword
               OR n.file_name CONTAINS $keyword
            RETURN n.name AS from, type(r) AS rel, m.name AS to
            LIMIT 10
        """, {"keyword": keyword})

        rows = result.data()
        if rows:
            for row in rows:
                print(f"  {row.get('from', '')} --[{row.get('rel', '')}]--> {row.get('to', '')}")
        else:
            print("  결과 없음")

    driver.close()


if __name__ == "__main__":
    # 1. 그래프 구축
    build_graph()

    # 2. 검색 테스트
    search_graph("졸업")
    search_graph("창업")
    search_graph("교과목")