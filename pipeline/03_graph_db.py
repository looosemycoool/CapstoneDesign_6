import os
import re
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EXTRACT_MODEL = "gpt-4o-mini"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_db(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[초기화] 기존 그래프 삭제 완료")


ALIAS_MAP = {
    "글솝": "글로벌소프트웨어융합전공",
    "글로벌소프트웨어": "글로벌소프트웨어융합전공",
    "다전공": "다전공프로그램",
    "부전공": "부전공프로그램",
    "조기졸업": "조기졸업요건",
    "조기 졸업": "조기졸업요건",
    "IT대학": "경북대학교IT대학",
    "컴퓨터학부": "경북대학교컴퓨터학부",
}

PLACEHOLDER_NAMES = {"홍길동"}


def normalize_node_name(text: str) -> str:
    if not text:
        return "Unknown"

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if text in ALIAS_MAP:
        return ALIAS_MAP[text]

    no_paren = re.sub(r"\(.*?\)", "", text).strip()
    no_paren = re.sub(r"\s+", " ", no_paren)

    if no_paren in ALIAS_MAP:
        return ALIAS_MAP[no_paren]

    return text if text else "Unknown"


def should_skip_entity_name(name: str) -> bool:
    name = normalize_node_name(str(name))
    return name in PLACEHOLDER_NAMES


def sanitize_label(label: str) -> str:
    if not label:
        return "Entity"

    cleaned = re.sub(r"[^\w]", "_", label.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    if cleaned and cleaned[0].isdigit():
        cleaned = "L_" + cleaned

    return cleaned if cleaned else "Entity"


def sanitize_relation_type(rel_type: str) -> str:
    if not rel_type:
        return "RELATED_TO"

    cleaned = re.sub(r"[^\w]", "_", rel_type.strip().upper(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    return cleaned if cleaned else "RELATED_TO"


def safe_json_load(text: str):
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


def extract_entities_and_relations(file_name: str, text: str) -> dict:
    prompt = f"""당신은 대학 행정 문서에서 지식 그래프를 구축하는 전문가입니다.
아래 경북대학교 컴퓨터학부 문서를 읽고, 의미 있는 개체(노드)와 개체 간 관계(엣지)를 추출하세요.

파일명: {file_name}
문서 내용:
{text[:3000]}

## 추출 규칙
- 스키마를 고정하지 말고, 문서 내용에서 자연스럽게 나타나는 개념을 개체 타입으로 사용하세요.
- 개체 타입은 짧고 명확한 영문 명사로 작성하세요.
- 비슷한 의미의 개체는 하나로 통합하세요.
- 관계 타입은 동사 형태의 영문 대문자로 작성하세요.
- name이 비어 있는 개체는 만들지 마세요.
- 홍길동처럼 예시용 이름이나 테스트용 placeholder 이름은 개체로 만들지 마세요.
- 확실하지 않은 관계는 억지로 만들지 마세요.
- relation의 from·to는 반드시 entities의 name 중 하나여야 합니다.

반드시 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력하세요.

{{
  "entities": [
    {{"name": "개체명", "type": "개체타입", "properties": {{}}}}
  ],
  "relations": [
    {{"from": "개체명1", "to": "개체명2", "type": "관계타입", "properties": {{}}}}
  ]
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=EXTRACT_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
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

        filtered_entities = []
        entity_names = set()

        for ent in entities:
            name = normalize_node_name(str(ent.get("name", "")))
            if not name or name == "Unknown" or should_skip_entity_name(name):
                continue

            ent["name"] = name
            filtered_entities.append(ent)
            entity_names.add(name)

        filtered_relations = []

        for rel in relations:
            from_name = normalize_node_name(str(rel.get("from", "")))
            to_name = normalize_node_name(str(rel.get("to", "")))

            if should_skip_entity_name(from_name) or should_skip_entity_name(to_name):
                continue

            if from_name in entity_names and to_name in entity_names:
                rel["from"] = from_name
                rel["to"] = to_name
                filtered_relations.append(rel)

        return {
            "entities": filtered_entities,
            "relations": filtered_relations,
        }

    except Exception as e:
        print(f"  [OpenAI 오류] {e}")
        return {"entities": [], "relations": []}


def consolidate_nodes(all_node_names: list[str]) -> dict[str, str]:
    all_node_names = [
        normalize_node_name(str(n))
        for n in all_node_names
        if n and normalize_node_name(str(n)) != "Unknown" and not should_skip_entity_name(str(n))
    ]

    all_node_names = list(dict.fromkeys(all_node_names))

    if len(all_node_names) <= 1:
        return {n: n for n in all_node_names}

    names_text = "\n".join(f"- {n}" for n in all_node_names)

    prompt = f"""아래는 지식 그래프에서 추출된 노드(개체) 이름 목록입니다.
의미가 동일하거나 매우 유사한 이름들을 하나의 대표 이름으로 통합하려 합니다.

노드 목록:
{names_text}

## 통합 규칙
- 맞춤법 차이, 띄어쓰기 차이, 동의어는 하나로 통합하세요.
- 의미가 다른 개체는 반드시 분리 유지하세요.
- 괄호 안 표현이 의미 구분에 중요하면 제거하지 말고 유지하세요.
- 대표 이름은 목록에 있는 이름 중 하나를 선택하거나, 더 명확한 표현이 있으면 새로 작성하세요.
- 통합하지 않아도 되는 노드도 결과에 반드시 포함하세요.
- 홍길동 같은 테스트용 이름은 결과에서 제외하세요.

반드시 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력하세요.

{{
  "원본이름1": "대표이름",
  "원본이름2": "대표이름"
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=EXTRACT_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.choices[0].message.content.strip()
        mapping = safe_json_load(result)

        normalized_mapping = {}

        for src, dst in mapping.items():
            src_norm = normalize_node_name(str(src))
            dst_norm = normalize_node_name(str(dst))

            if should_skip_entity_name(src_norm) or should_skip_entity_name(dst_norm):
                continue

            if src_norm and src_norm != "Unknown" and dst_norm and dst_norm != "Unknown":
                normalized_mapping[src_norm] = dst_norm

        for name in all_node_names:
            name_norm = normalize_node_name(str(name))
            if (
                name_norm
                and name_norm != "Unknown"
                and not should_skip_entity_name(name_norm)
                and name_norm not in normalized_mapping
            ):
                normalized_mapping[name_norm] = name_norm

        return normalized_mapping

    except Exception as e:
        print(f"  [노드 통합 오류] {e}")
        return {n: n for n in all_node_names}


def create_document_node(session, file_name: str):
    session.run(
        """
        MERGE (d:Document {file_name: $file_name})
        SET d.name = $file_name,
            d.title = $file_name
        """,
        {"file_name": file_name},
    )


def upsert_entity(session, name: str, etype: str, props: dict) -> str | None:
    name = normalize_node_name(str(name))

    if not name or name == "Unknown" or should_skip_entity_name(name):
        return None

    label = sanitize_label(etype or "Entity")

    safe_props = {}
    for k, v in (props or {}).items():
        if not k:
            continue

        key = re.sub(r"[^\w]", "_", str(k), flags=re.UNICODE).strip("_")
        if key:
            safe_props[key] = "" if v is None else str(v)

    session.run(
        f"""
        MERGE (e:Entity {{name: $name}})
        SET e += $props
        SET e:`{label}`
        """,
        {"name": name, "props": safe_props},
    )

    return name


def create_relation(session, from_name: str, to_name: str, rel_type: str):
    from_name = normalize_node_name(str(from_name))
    to_name = normalize_node_name(str(to_name))

    if (
        not from_name
        or not to_name
        or from_name == "Unknown"
        or to_name == "Unknown"
        or should_skip_entity_name(from_name)
        or should_skip_entity_name(to_name)
    ):
        return

    rel = sanitize_relation_type(rel_type)

    session.run(
        f"""
        MATCH (a:Entity {{name: $from_name}})
        MATCH (b:Entity {{name: $to_name}})
        MERGE (a)-[:`{rel}`]->(b)
        """,
        {"from_name": from_name, "to_name": to_name},
    )


def build_graph():
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")

    if not os.path.exists(manual_path):
        print(f"[오류] 파일이 없습니다: {manual_path}")
        return

    with open(manual_path, encoding="utf-8") as f:
        manual_files = json.load(f)

    driver = get_driver()

    try:
        clear_db(driver)

        print(f"[시작] Graph DB 구축 ({len(manual_files)}개 파일) — GPT-4o-mini 동적 추출 모드")

        all_entity_names = []
        extraction_results = []

        for i, mf in enumerate(manual_files):
            file_name = mf.get("file_name", f"unknown_{i}")
            parsed_text = str(mf.get("parsed_text", "")).strip()

            if not parsed_text:
                print(f"[{i + 1}/{len(manual_files)}] {file_name[:40]} → 텍스트 없음 스킵")
                continue

            print(f"\n[{i + 1}/{len(manual_files)}] {file_name[:60]}")

            extracted = extract_entities_and_relations(file_name, parsed_text)

            entities_count = len(extracted.get("entities", []))
            relations_count = len(extracted.get("relations", []))

            print(f"  개체: {entities_count}개, 관계: {relations_count}개 추출")

            extraction_results.append((file_name, extracted))

            for ent in extracted.get("entities", []):
                name = normalize_node_name(str(ent.get("name", "")))

                if name and name != "Unknown" and not should_skip_entity_name(name):
                    all_entity_names.append(name)

        unique_names = list(dict.fromkeys(all_entity_names))

        print(f"\n[노드 통합] 고유 개체명 {len(unique_names)}개 → GPT-4o-mini 유사도 분석 중...")

        BATCH = 80
        name_mapping: dict[str, str] = {}

        for start in range(0, len(unique_names), BATCH):
            batch_names = unique_names[start:start + BATCH]
            batch_map = consolidate_nodes(batch_names)
            name_mapping.update(batch_map)
            print(f"  배치 {start // BATCH + 1}: {len(batch_names)}개 처리 완료")

        representatives = list(dict.fromkeys(name_mapping.values()))

        if len(representatives) > 1:
            print("  배치 대표 이름 추가 통합 중...")
            second_pass_map = consolidate_nodes(representatives)

            for src, dst in list(name_mapping.items()):
                name_mapping[src] = second_pass_map.get(dst, dst)

        print("\n[저장] Neo4j에 노드·관계 저장 중...")

        with driver.session() as session:
            for file_name, extracted in extraction_results:
                create_document_node(session, file_name)

                valid_names = set()

                for ent in extracted.get("entities", []):
                    raw_name = normalize_node_name(str(ent.get("name", "")))

                    if (
                        not raw_name
                        or raw_name == "Unknown"
                        or should_skip_entity_name(raw_name)
                    ):
                        continue

                    canonical = normalize_node_name(name_mapping.get(raw_name, raw_name))

                    if (
                        not canonical
                        or canonical == "Unknown"
                        or should_skip_entity_name(canonical)
                    ):
                        continue

                    created = upsert_entity(
                        session,
                        canonical,
                        ent.get("type", "Entity"),
                        ent.get("properties", {}),
                    )

                    if created:
                        valid_names.add(canonical)

                        session.run(
                            """
                            MATCH (d:Document {file_name: $file_name})
                            MATCH (e:Entity {name: $name})
                            MERGE (d)-[:MENTIONS]->(e)
                            """,
                            {"file_name": file_name, "name": canonical},
                        )

                for rel in extracted.get("relations", []):
                    from_raw = normalize_node_name(str(rel.get("from", "")))
                    to_raw = normalize_node_name(str(rel.get("to", "")))
                    rel_type = rel.get("type", "RELATED_TO")

                    if (
                        not from_raw
                        or not to_raw
                        or from_raw == "Unknown"
                        or to_raw == "Unknown"
                        or should_skip_entity_name(from_raw)
                        or should_skip_entity_name(to_raw)
                    ):
                        continue

                    from_name = normalize_node_name(name_mapping.get(from_raw, from_raw))
                    to_name = normalize_node_name(name_mapping.get(to_raw, to_raw))

                    if from_name not in valid_names or to_name not in valid_names:
                        continue

                    create_relation(session, from_name, to_name, rel_type)

            node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

        print("\n[완료] Graph DB 구축 완료!")
        print(f"총 노드: {node_count}개")
        print(f"총 관계: {rel_count}개")
        print("\n[확인] Neo4j 웹 브라우저에서 시각적 확인:")
        print("  → http://localhost:7474")
        print("  → 쿼리: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")

    finally:
        driver.close()


def search_graph(keyword: str):
    driver = get_driver()

    try:
        print(f"\n[그래프 검색] '{keyword}'")

        with driver.session() as session:
            result = session.run(
                """
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
                """,
                {"keyword": keyword},
            )

            rows = result.data()

            if rows:
                for row in rows:
                    print(f"  {row.get('from_node', '')} --[{row.get('rel', '')}]--> {row.get('to_node', '')}")
            else:
                print("  결과 없음")

    finally:
        driver.close()


def check_duplicate_nodes():
    driver = get_driver()

    try:
        print("\n[중복 노드 검사]")

        with driver.session() as session:
            rows = session.run(
                """
                MATCH (n:Entity)
                WHERE n.name IS NOT NULL
                WITH n.name AS name, count(n) AS cnt
                WHERE cnt > 1
                RETURN name, cnt
                ORDER BY cnt DESC
                LIMIT 20
                """
            ).data()

            if rows:
                print("중복 노드 발견:")
                for row in rows:
                    print(f"  {row['name']}: {row['cnt']}개")
            else:
                print("동일 name 기준 중복 노드 없음")

    finally:
        driver.close()


if __name__ == "__main__":
    build_graph()
    check_duplicate_nodes()
    search_graph("졸업")
    search_graph("장학")
    search_graph("교과목")
