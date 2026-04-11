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

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EXTRACT_MODEL = "gpt-4o-mini"   # 개체·관계 추출 및 노드 통합에 사용


# ── Neo4j 드라이버 ────────────────────────────────────────
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_db(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[초기화] 기존 그래프 삭제 완료")


# ── 문자열 정제 ───────────────────────────────────────────
def sanitize_label(label: str) -> str:
    """Neo4j 노드 라벨로 사용 가능한 형태로 정제 (알파벳/숫자/언더스코어만)"""
    if not label:
        return "Entity"
    # 한글·알파벳·숫자·언더스코어만 허용
    cleaned = re.sub(r"[^\w]", "_", label.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    # Neo4j 라벨은 숫자로 시작 불가
    if cleaned and cleaned[0].isdigit():
        cleaned = "L_" + cleaned
    return cleaned if cleaned else "Entity"


def sanitize_relation_type(rel_type: str) -> str:
    """Neo4j 관계 타입으로 사용 가능한 형태로 정제 (대문자+언더스코어)"""
    if not rel_type:
        return "RELATED_TO"
    cleaned = re.sub(r"[^\w]", "_", rel_type.strip().upper(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned if cleaned else "RELATED_TO"


# ── JSON 안전 파싱 ────────────────────────────────────────
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


# ── Step 1: 개체·관계 자동 추출 (GPT-4o-mini, 동적 스키마) ──────
def extract_entities_and_relations(file_name: str, text: str) -> dict:
    """GPT-4o-mini를 사용해 문서에서 개체·관계를 동적으로 추출 (스키마 고정 없음)"""
    prompt = f"""당신은 대학 행정 문서에서 지식 그래프를 구축하는 전문가입니다.
아래 경북대학교 컴퓨터학부 문서를 읽고, 의미 있는 개체(노드)와 개체 간 관계(엣지)를 추출하세요.

파일명: {file_name}
문서 내용:
{text[:3000]}

## 추출 규칙
- 스키마를 고정하지 말고, 문서 내용에서 자연스럽게 나타나는 개념을 개체 타입으로 사용하세요.
- 개체 타입은 짧고 명확한 영문 명사로 작성하세요 (예: Scholarship, GraduationRequirement, Course, Deadline, Department, Program, Student, Condition, Schedule, Rule).
- 비슷한 의미의 개체는 하나로 통합하세요 (예: "졸업요건"과 "졸업 요건"은 동일 개체).
- 관계 타입은 동사 형태의 영문 대문자로 작성하세요 (예: REQUIRES, HAS_DEADLINE, PART_OF, APPLIES_TO, RELATED_TO, HAS_CONDITION, BELONGS_TO, OFFERS, TARGETS).
- name이 비어 있는 개체는 만들지 마세요.
- 확실하지 않은 관계는 억지로 만들지 마세요.
- relation의 from·to는 반드시 entities의 name 중 하나여야 합니다.

[Few-shot 예시]
문서: "2026학년도 소프트웨어융합전공 산학프로젝트(3학점) 결과보고서 제출 안내. 기한은 6월 15일까지이며, 우수팀에게는 SW교육원에서 50만원의 장학금을 지급합니다."
추출 논리:
- Entity 1: name="소프트웨어융합전공", type="Major"
- Entity 2: name="산학프로젝트", type="Course", properties=[(key="credits", value="3학점"), (key="deadline", value="6월 15일")]
- Entity 3: name="SW교육원", type="Organization"
- Entity 4: name="우수팀 장학금", type="Funding", properties=[(key="amount", value="50만원")]
- Relation 1: from="소프트웨어융합전공", to="산학프로젝트", type="REQUIRES"
- Relation 2: from="SW교육원", to="우수팀 장학금", type="PROVIDES"
- Relation 3: from="우수팀 장학금", to="산학프로젝트", type="REWARDS"

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
            messages=[{"role": "user", "content": prompt}]
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
        print(f"  [OpenAI 오류] {e}")
        return {"entities": [], "relations": []}


# ── Step 2: 유사 노드 통합 (GPT-4o-mini) ──────────────────────
def consolidate_nodes(all_node_names: list[str]) -> dict[str, str]:
    """전체 노드 목록에서 의미가 유사한 노드를 하나의 대표 이름으로 통합.
    반환: {원본 이름 → 대표 이름} 매핑"""
    if len(all_node_names) <= 1:
        return {n: n for n in all_node_names}

    names_text = "\n".join(f"- {n}" for n in all_node_names)

    prompt = f"""아래는 지식 그래프에서 추출된 노드(개체) 이름 목록입니다.
의미가 동일하거나 매우 유사한 이름들을 하나의 대표 이름으로 통합하려 합니다.

노드 목록:
{names_text}

## 통합 규칙
- 맞춤법 차이, 띄어쓰기 차이, 동의어는 하나로 통합하세요.
  예: "졸업요건", "졸업 요건", "졸업이수요건" → 가장 공식적인 표현 하나로 통합
- 의미가 다른 개체는 반드시 분리 유지하세요.
- 대표 이름은 목록에 있는 이름 중 하나를 선택하거나, 더 명확한 표현이 있으면 새로 작성하세요.
- 통합하지 않아도 되는 노드도 결과에 반드시 포함하세요 (자기 자신이 대표).

반드시 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력하세요.
키: 원본 노드 이름, 값: 통합될 대표 노드 이름

{{
  "원본이름1": "대표이름",
  "원본이름2": "대표이름",
  ...
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=EXTRACT_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()

        # JSON 추출
        text = result
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
        if start != -1 and end != -1:
            text = text[start:end + 1]

        mapping = json.loads(text)

        # 누락된 노드는 자기 자신으로 매핑
        for name in all_node_names:
            if name not in mapping:
                mapping[name] = name

        return mapping

    except Exception as e:
        print(f"  [노드 통합 오류] {e}")
        return {n: n for n in all_node_names}


# ── Neo4j 저장 ───────────────────────────────────────────
def create_document_node(session, file_name: str):
    session.run("""
        MERGE (d:Document {file_name: $file_name})
        SET d.name = $file_name, d.title = $file_name
    """, {"file_name": file_name})


def upsert_entity(session, name: str, etype: str, props: dict) -> str | None:
    name = str(name).strip()
    if not name:
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
        f"MERGE (e:`{label}` {{name: $name}}) SET e += $props",
        {"name": name, "props": safe_props}
    )
    return name


def create_relation(session, from_name: str, to_name: str, rel_type: str):
    rel = sanitize_relation_type(rel_type)
    session.run(
        f"""
        MATCH (a {{name: $from_name}})
        MATCH (b {{name: $to_name}})
        MERGE (a)-[:`{rel}`]->(b)
        """,
        {"from_name": from_name, "to_name": to_name}
    )


# ── 전체 그래프 구축 ─────────────────────────────────────
def build_graph():
    """전체 그래프 구축
    1단계: GPT-4o-mini로 각 문서에서 개체·관계 동적 추출
    2단계: GPT-4o-mini로 전체 노드에서 유사 노드 통합 후 Neo4j 저장
    """
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")
    if not os.path.exists(manual_path):
        print(f"[오류] 파일이 없습니다: {manual_path}")
        return

    with open(manual_path, encoding="utf-8") as f:
        manual_files = json.load(f)

    driver = get_driver()
    clear_db(driver)

    print(f"[시작] Graph DB 구축 ({len(manual_files)}개 파일) — GPT-4o-mini 동적 추출 모드")

    # ── 1단계: 문서별 개체·관계 추출 및 저장 ──────────────
    all_entity_names = []       # 통합 대상 수집용
    extraction_results = []     # (file_name, extracted) 캐시

    for i, mf in enumerate(manual_files):
        file_name   = mf.get("file_name", f"unknown_{i}")
        parsed_text = str(mf.get("parsed_text", "")).strip()

        if not parsed_text:
            print(f"[{i+1}/{len(manual_files)}] {file_name[:40]} → 텍스트 없음 스킵")
            continue

        print(f"\n[{i+1}/{len(manual_files)}] {file_name[:60]}")

        extracted = extract_entities_and_relations(file_name, parsed_text)
        entities_count  = len(extracted.get("entities", []))
        relations_count = len(extracted.get("relations", []))
        print(f"  개체: {entities_count}개, 관계: {relations_count}개 추출")

        extraction_results.append((file_name, extracted))

        # 개체 이름 수집 (노드 통합 단계에서 사용)
        for ent in extracted.get("entities", []):
            name = str(ent.get("name", "")).strip()
            if name:
                all_entity_names.append(name)

    # ── 2단계: 전체 노드 이름 중복 제거 후 유사 노드 통합 ──
    unique_names = list(dict.fromkeys(all_entity_names))  # 순서 유지 중복 제거
    print(f"\n[노드 통합] 고유 개체명 {len(unique_names)}개 → GPT-4o-mini 유사도 분석 중...")

    # 노드가 많으면 배치로 나눠 처리
    BATCH = 80
    name_mapping: dict[str, str] = {}
    for start in range(0, len(unique_names), BATCH):
        batch_names = unique_names[start:start + BATCH]
        batch_map = consolidate_nodes(batch_names)
        name_mapping.update(batch_map)
        print(f"  배치 {start//BATCH + 1}: {len(batch_names)}개 처리 완료")

    # ── 3단계: 매핑 적용 후 Neo4j 저장 ────────────────────
    print("\n[저장] Neo4j에 노드·관계 저장 중...")

    with driver.session() as session:
        for file_name, extracted in extraction_results:
            create_document_node(session, file_name)

            valid_names = set()

            for entity in extracted.get("entities", []):
                raw_name = str(entity.get("name", "")).strip()
                if not raw_name:
                    continue
                # 통합된 대표 이름으로 교체
                canonical = name_mapping.get(raw_name, raw_name)
                created = upsert_entity(session, canonical, entity.get("type", "Entity"), entity.get("properties", {}))
                if created:
                    valid_names.add(canonical)

                    session.run("""
                        MATCH (d:Document {file_name: $file_name})
                        MATCH (e {name: $name})
                        MERGE (d)-[:MENTIONS]->(e)
                    """, {"file_name": file_name, "name": canonical})

            for rel in extracted.get("relations", []):
                from_raw = str(rel.get("from", "")).strip()
                to_raw   = str(rel.get("to", "")).strip()
                rel_type = rel.get("type", "RELATED_TO")

                if not from_raw or not to_raw:
                    continue

                from_name = name_mapping.get(from_raw, from_raw)
                to_name   = name_mapping.get(to_raw, to_raw)

                if from_name not in valid_names or to_name not in valid_names:
                    continue

                create_relation(session, from_name, to_name, rel_type)

        # ── 최종 통계 ─────────────────────────────────────
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

    print("\n[완료] Graph DB 구축 완료!")
    print(f"총 노드: {node_count}개")
    print(f"총 관계: {rel_count}개")
    print("\n[확인] Neo4j 웹 브라우저에서 시각적 확인:")
    print("  → http://localhost:7474")
    print("  → 쿼리: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")

    driver.close()


# ── 그래프 검색 테스트 ────────────────────────────────────
def search_graph(keyword: str):
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
    search_graph("장학")
    search_graph("교과목")
