import os
import re
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# ── 경로 / 환경변수 ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# 허용 타입
ALLOWED_ENTITY_TYPES = {
    "Document", "Scholarship", "Schedule", "Requirement",
    "Department", "Program", "Course", "Student", "Entity"
}

ALLOWED_RELATION_TYPES = {
    "HAS_DEADLINE", "REQUIRES", "RELATED_TO",
    "PART_OF", "APPLIES_TO", "HAS_CONDITION", "MENTIONS"
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_db(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[초기화] 기존 그래프 삭제 완료")


def sanitize_label(label: str) -> str:
    """Neo4j 라벨 안전하게 정제"""
    if not label:
        return "Entity"
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", label.strip())
    if not cleaned:
        return "Entity"
    if cleaned not in ALLOWED_ENTITY_TYPES:
        return "Entity"
    return cleaned


def sanitize_relation_type(rel_type: str) -> str:
    """Neo4j 관계 타입 안전하게 정제"""
    if not rel_type:
        return "RELATED_TO"
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", rel_type.strip().upper())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "RELATED_TO"
    if cleaned not in ALLOWED_RELATION_TYPES:
        return "RELATED_TO"
    return cleaned


def safe_json_load(text: str):
    """LLM 응답에서 JSON 안전 추출"""
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start:end + 1]

    return json.loads(text)


def extract_entities_and_relations(file_name, text):
    """LLM으로 파일에서 개체·관계 추출"""
    prompt = f"""다음 경북대학교 컴퓨터학부 문서에서 개체와 관계를 추출해주세요.

파일명: {file_name}
내용:
{text[:3000]}

반드시 아래 JSON 형식으로만 응답하세요. 설명 문장 없이 JSON만 출력하세요.

{{
  "entities": [
    {{
      "name": "개체명",
      "type": "개체타입",
      "properties": {{}}
    }}
  ],
  "relations": [
    {{
      "from": "개체명1",
      "to": "개체명2",
      "type": "관계타입",
      "properties": {{}}
    }}
  ]
}}

개체 타입 후보:
Document, Scholarship, Schedule, Requirement, Department, Program, Course, Student

관계 타입 후보:
HAS_DEADLINE, REQUIRES, RELATED_TO, PART_OF, APPLIES_TO, HAS_CONDITION

주의:
- name이 비어 있으면 안 됩니다.
- relation의 from, to는 반드시 entities의 name과 연결되게 작성하세요.
- 확실하지 않으면 개체/관계를 억지로 만들지 마세요.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        parsed = safe_json_load(result)

        if not isinstance(parsed, dict):
            return {"entities": [], "relations": []}

        entities = parsed.get("entities", [])
        relations = parsed.get("relations", [])

        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []

        return {"entities": entities, "relations": relations}

    except Exception as e:
        print(f"  [LLM 오류] {e}")
        return {"entities": [], "relations": []}


def create_document_node(session, file_name):
    """문서 노드 생성"""
    session.run("""
        MERGE (d:Document {file_name: $file_name})
        SET d.file_name = $file_name,
            d.name = $file_name,
            d.title = $file_name
    """, {"file_name": file_name})


def create_entity_node(session, entity):
    """개체 노드 생성"""
    name = str(entity.get("name", "")).strip()
    if not name:
        return None

    etype = sanitize_label(entity.get("type", "Entity"))
    props = entity.get("properties", {})
    if not isinstance(props, dict):
        props = {}

    # Neo4j에 넣기 안전한 문자열 속성만 저장
    safe_props = {}
    for k, v in props.items():
        if not k:
            continue
        key = re.sub(r"[^A-Za-z0-9_]", "_", str(k)).strip("_")
        if not key:
            continue
        safe_props[key] = "" if v is None else str(v)

    query = f"""
        MERGE (e:{etype} {{name: $name}})
        SET e.name = $name
        SET e += $props
        RETURN e.name AS name
    """
    session.run(query, {"name": name, "props": safe_props})
    return name


def create_entities_and_relations(session, file_name, extracted):
    """추출된 개체·관계를 Neo4j에 저장"""
    valid_names = set()

    for entity in extracted.get("entities", []):
        created_name = create_entity_node(session, entity)
        if not created_name:
            continue

        valid_names.add(created_name)

        session.run("""
            MATCH (d:Document {file_name: $file_name})
            MATCH (e {name: $name})
            MERGE (d)-[:MENTIONS]->(e)
        """, {"file_name": file_name, "name": created_name})

    for rel in extracted.get("relations", []):
        from_name = str(rel.get("from", "")).strip()
        to_name = str(rel.get("to", "")).strip()
        rel_type = sanitize_relation_type(rel.get("type", "RELATED_TO"))

        if not from_name or not to_name:
            continue

        if from_name not in valid_names or to_name not in valid_names:
            continue

        session.run(f"""
            MATCH (a {{name: $from_name}})
            MATCH (b {{name: $to_name}})
            MERGE (a)-[r:{rel_type}]->(b)
        """, {"from_name": from_name, "to_name": to_name})


def build_graph():
    """전체 그래프 구축 - manual_parsed.json 사용"""
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")
    if not os.path.exists(manual_path):
        print(f"[오류] 파일이 없습니다: {manual_path}")
        return

    with open(manual_path, encoding="utf-8") as f:
        manual_files = json.load(f)

    driver = get_driver()
    clear_db(driver)

    print(f"[시작] Graph DB 구축 ({len(manual_files)}개 파일)")

    with driver.session() as session:
        for i, mf in enumerate(manual_files):
            file_name = mf.get("file_name", f"unknown_{i}")
            parsed_text = str(mf.get("parsed_text", "")).strip()

            if not parsed_text:
                print(f"[{i+1}/{len(manual_files)}] {file_name[:40]} → 텍스트 없음 스킵")
                continue

            print(f"\n[{i+1}/{len(manual_files)}] {file_name[:60]}")

            create_document_node(session, file_name)

            extracted = extract_entities_and_relations(file_name, parsed_text)
            entities_count = len(extracted.get("entities", []))
            relations_count = len(extracted.get("relations", []))
            print(f"  개체: {entities_count}개, 관계: {relations_count}개 추출")

            create_entities_and_relations(session, file_name, extracted)

    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

    print("\n[완료] Graph DB 구축 완료!")
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
            WHERE coalesce(n.name, '') CONTAINS $keyword
               OR coalesce(n.title, '') CONTAINS $keyword
               OR coalesce(n.file_name, '') CONTAINS $keyword
               OR coalesce(m.name, '') CONTAINS $keyword
               OR coalesce(m.title, '') CONTAINS $keyword
               OR coalesce(m.file_name, '') CONTAINS $keyword
            RETURN
                coalesce(n.name, n.title, n.file_name, '') AS from_node,
                type(r) AS rel,
                coalesce(m.name, m.title, m.file_name, '') AS to_node
            LIMIT 10
        """, {"keyword": keyword})

        rows = result.data()
        if rows:
            for row in rows:
                print(f"  {row.get('from_node', '')} --[{row.get('rel', '')}]--> {row.get('to_node', '')}")
        else:
            print("  결과 없음")

    driver.close()


if __name__ == "__main__":
    build_graph()
    search_graph("졸업")
    search_graph("창업")
    search_graph("교과목")