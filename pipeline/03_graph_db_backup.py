import os
import re
import sys
import json
import traceback
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Windows 콘솔(cp949) 에서도 한글 / em-dash 등 유니코드 출력 안전하게.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
EXTRACT_MODEL = "solar-pro3"   # 개체·관계 추출 및 노드 통합에 사용

# 그래프 추출용 청킹 설정.
# GraphRAG 연구: 600토큰 청크가 2400토큰보다 2배 더 많은 엔티티 추출.
# 한국어 기준 1자 ≈ 0.4~0.6 토큰 → 1500자 ≈ 600~900 토큰.
GRAPH_CHUNK_SIZE = 1500
GRAPH_CHUNK_OVERLAP = 150

# 노드 통합 배치 크기 — 1차/2차 모두 동일하게 적용해 프롬프트 과부하 방지.
CONSOLIDATE_BATCH = 80


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

_NULL_LIKE_NAMES = {"none", "null", "n/a", "nan", "undefined", "unknown", ""}


def normalize_node_name(text) -> str:
    """노드 이름 정규화. None/null/비정형 입력은 'Unknown' 으로 거부."""
    if text is None:
        return "Unknown"

    if not isinstance(text, (str, int, float)):
        return "Unknown"

    text = str(text).strip()
    if text.lower() in _NULL_LIKE_NAMES:
        return "Unknown"

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


# ── 청킹 ────────────────────────────────────────────────────
_graph_splitter = RecursiveCharacterTextSplitter(
    chunk_size=GRAPH_CHUNK_SIZE,
    chunk_overlap=GRAPH_CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n## ", "\n# ",
        "\n가. ", "\n나. ", "\n다. ", "\n라. ", "\n마. ",
        "\n1) ", "\n2) ", "\n3) ", "\n4) ", "\n5) ",
        "\n① ", "\n② ", "\n③ ", "\n④ ", "\n⑤ ",
        "\n- ", "\n* ",
        "\n", ". ", " ", "",
    ],
    length_function=len,
)


def split_into_chunks(text: str) -> list[str]:
    """문서를 GRAPH_CHUNK_SIZE 단위로 분할. 1청크면 리스트 1개 반환."""
    chunks = _graph_splitter.split_text(text)
    return chunks if chunks else [text]


def merge_chunk_extractions(chunk_results: list[dict]) -> dict:
    """청크별 추출 결과를 하나의 문서 단위 결과로 병합.

    - 같은 이름의 엔티티: properties 누적 병합, 타입은 첫 등장 기준 유지.
    - 관계: (from, to, type) 기준 중복 제거.
    """
    merged_entities: dict[str, dict] = {}  # name → entity dict
    merged_relations: list[dict] = []
    seen_relations: set[tuple] = set()

    for result in chunk_results:
        for ent in result.get("entities", []):
            name = ent.get("name", "")
            if not name:
                continue
            if name not in merged_entities:
                merged_entities[name] = {
                    "name": name,
                    "type": ent.get("type", "Entity"),
                    "properties": dict(ent.get("properties") or {}),
                }
            else:
                # 같은 엔티티가 다른 청크에서 추가 properties 제공할 수 있으므로 병합
                existing_props = merged_entities[name]["properties"]
                for k, v in (ent.get("properties") or {}).items():
                    if k not in existing_props:
                        existing_props[k] = v

        for rel in result.get("relations", []):
            key = (rel.get("from", ""), rel.get("to", ""), rel.get("type", ""))
            if key not in seen_relations:
                seen_relations.add(key)
                merged_relations.append(rel)

    return {
        "entities": list(merged_entities.values()),
        "relations": merged_relations,
    }


# ── Step 1: 개체·관계 자동 추출 (solar-pro, 동적 스키마) ──────
def extract_entities_and_relations(file_name: str, text: str) -> dict:
    """solar-pro를 사용해 청크 텍스트에서 개체·관계를 동적으로 추출.

    청크 단위로 호출되므로 트런케이션 없이 전체 텍스트를 입력한다.
    """

    prompt = f"""당신은 대학 행정 문서에서 지식 그래프를 구축하는 전문가입니다.
아래 경북대학교 문서를 읽고, 의미 있는 개체(노드)와 개체 간 관계(엣지)를 추출하세요.

파일명: {file_name}
문서 내용:
{text}

## 추출 규칙
- 스키마를 고정하지 말고, 문서 내용에서 자연스럽게 나타나는 개념을 개체 타입으로 사용하세요.
- **개체 이름(name)은 반드시 한국어로 작성하세요. 영어 단어/캐멀케이스(GraduationRequirement, InternshipCredit 등) 절대 금지.** 문서에 한국어 표현이 있다면 그대로 사용하세요. KUCIS 같은 약어는 한글 풀네임("대학정보보호동아리") 우선, 약어 자체가 표준이면 약어 유지.
- 개체 타입(type)은 영문 명사 (예: Major, Course, Organization, Funding, Requirement, Condition, Period).
- 미리 정해진 타입 목록은 없습니다. 문서 내용에 가장 잘 맞는 타입을 스스로 선택하세요.
- 비슷한 의미의 개체는 하나로 통합하세요 (예: "졸업요건"과 "졸업 요건"은 동일 개체).
- 날짜, 금액, URL, 학점, 점수 등 단순 수치 정보는 'properties' 에 key-value 로 저장하세요. 단, **임계값으로 작용하는 수치(평점 1.7 미만, 토익 800점 이상)는 별도 노드(예: "평점 1.7 미만 학생")로 만들고 HAS_THRESHOLD 관계로 연결**하세요.
- name이 비어 있는 개체는 만들지 마세요.
- 홍길동처럼 예시용 이름이나 테스트용 placeholder 이름은 개체로 만들지 마세요.
- 확실하지 않은 관계는 억지로 만들지 마세요.
- relation의 from·to는 반드시 entities의 name 중 하나여야 합니다.

## 관계 타입 (영문 대문자, 동사형). 다음 17개를 우선 사용:
1. **REQUIRES** — A 가 B 를 요구함 (필수 조건)
2. **HAS_CONDITION** — A 의 일반 조건이 B
3. **HAS_THRESHOLD** — A 가 B(수치 임계) 를 기준으로 함. 예: 학사경고 HAS_THRESHOLD 평점 1.7 미만
4. **HAS_DEADLINE** — A 의 마감 기한이 B
5. **TARGETS** — A 의 대상이 B
6. **OFFERS** / **PROVIDES** — A 가 B 를 제공
7. **INCLUDES** — A 가 B 를 포함 (전체-부분)
8. **PART_OF** / **BELONGS_TO** — B 가 A 의 구성 요소 / 소속
9. **APPLIES_TO** — A 가 B 에 적용됨
10. **EXCLUDES** — A 가 B 를 제외
11. **EXCLUDES_FROM** — A 가 B(상위 집합)에서 제외됨. 예: 군휴학 EXCLUDES_FROM 학사경고대상
12. **HAS_EXCEPTION** — A 의 예외 규정이 B. 예: 졸업요건 HAS_EXCEPTION 외국인유학생
13. **SUBSTITUTES_FOR** — A 가 B 를 대체. 예: 논문게재 SUBSTITUTES_FOR 현장실습 / 자격증 SUBSTITUTES_FOR 어학시험
14. **ALTERNATIVE_PATH** — A 와 B 가 동등한 경로 (택일 가능). 예: 캡스톤설계 ALTERNATIVE_PATH 종합설계
15. **REWARDS** / **CHARGES** — 보상 / 비용 부과
16. **ACCEPTS** / **REFERS** — 수용 / 참조
17. **RELATED_TO** — 위에 안 맞을 때만 사용 (남용 금지)

**lateral 관계(11~14)가 조건/대체 추론에 결정적이니 적극 추출하세요.** 단순 부모-자식만 추출하면 그래프가 빈약합니다.

[Few-shot 예시]
문서: "2026학년도 소프트웨어융합전공 산학프로젝트(3학점) 결과보고서 제출 안내. 기한은 6월 15일까지. 학사 경고는 평점평균 1.7 미만 학생에게 적용되며, 외국인유학생은 학사경고 적용에서 제외됩니다. 학·석사연계 트랙에서는 KCI 등재 학술지에 주저자 논문 게재 시 현장실습 3학점을 대체합니다."
추출 논리:
- Entity 1: name="소프트웨어융합전공", type="Major"
- Entity 2: name="산학프로젝트", type="Course", properties={{"credits":"3학점","deadline":"6월 15일"}}
- Entity 3: name="학사경고", type="Status"
- Entity 4: name="평점 1.7 미만 학생", type="Threshold"
- Entity 5: name="외국인유학생", type="Cohort"
- Entity 6: name="현장실습", type="Requirement", properties={{"credits":"3학점"}}
- Entity 7: name="학·석사연계 트랙", type="Track"
- Entity 8: name="KCI 주저자 논문게재", type="Achievement"
- Relation 1: from="소프트웨어융합전공", to="산학프로젝트", type="REQUIRES"
- Relation 2: from="학사경고", to="평점 1.7 미만 학생", type="HAS_THRESHOLD"
- Relation 3: from="학사경고", to="외국인유학생", type="HAS_EXCEPTION"
- Relation 4: from="외국인유학생", to="학사경고", type="EXCLUDES_FROM"
- Relation 5: from="KCI 주저자 논문게재", to="현장실습", type="SUBSTITUTES_FOR"
- Relation 6: from="학·석사연계 트랙", to="현장실습", type="REQUIRES"

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
        response = upstage_client.chat.completions.create(
            model=EXTRACT_MODEL,
            max_tokens=4096,
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
            if not isinstance(ent, dict):
                continue
            name = normalize_node_name(ent.get("name"))
            if not name or name == "Unknown" or should_skip_entity_name(name):
                continue

            ent["name"] = name
            filtered_entities.append(ent)
            entity_names.add(name)

        filtered_relations = []

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            from_name = normalize_node_name(rel.get("from"))
            to_name = normalize_node_name(rel.get("to"))

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
        print(f"  [추출 오류] {file_name}: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return {"entities": [], "relations": []}


# ── Step 2: 유사 노드 통합 (solar-pro) ──────────────────────
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
- 맞춤법 차이, 띄어쓰기 차이, 동의어는 하나로 통합하세요 (예: "복학생", "복학 학생", "복학 대상자" → "복학").
- **영어 캐멀케이스 노드(GraduationRequirement, InternshipCredit 등)는 한국어로 번역 후 통합**하세요. 예: GraduationRequirement → 졸업요건, InternshipCredit → 현장실습 학점, ResearchPaperPublication → 논문게재.
- 같은 개념이 영어/한국어로 둘 다 있으면 **한국어 표현을 대표 이름으로** 사용하세요.
- 의미가 다른 개체는 반드시 분리 유지하세요.
- 괄호 안 표현이 의미 구분에 중요하면 제거하지 말고 유지하세요.
- 대표 이름은 목록에 있는 이름 중 하나를 선택하거나, 더 명확한 한국어 표현이 있으면 새로 작성하세요.
- 통합하지 않아도 되는 노드도 결과에 반드시 포함하세요.
- 홍길동 같은 테스트용 이름은 결과에서 제외하세요.

반드시 아래 JSON 형식으로만 응답하세요. 설명 없이 JSON만 출력하세요.

{{
  "원본이름1": "대표이름",
  "원본이름2": "대표이름"
}}"""

    try:
        response = upstage_client.chat.completions.create(
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


# ── Neo4j 저장 ───────────────────────────────────────────
def create_document_node(session, file_name: str):
    session.run(
        """
        MERGE (d:Document {file_name: $file_name})
        SET d.name = $file_name,
            d.title = $file_name
        """,
        {"file_name": file_name},
    )


def upsert_entity(session, name: str, etype: str, props, file_name: str = "") -> str | None:
    name = normalize_node_name(str(name))

    if not name or name == "Unknown" or should_skip_entity_name(name):
        return None

    label = sanitize_label(etype or "Entity")
    safe_props = {}

    if isinstance(props, dict):
        raw_items = props.items()
    elif isinstance(props, list):
        raw_items = []
        for item in props:
            if isinstance(item, dict):
                if "key" in item and "value" in item:
                    raw_items.append((item["key"], item["value"]))
                else:
                    raw_items.extend(item.items())
    else:
        raw_items = []

    for k, v in raw_items:
        if not k:
            continue

        key = re.sub(r"[^\w]", "_", str(k), flags=re.UNICODE).strip("_")
        if key:
            safe_props[key] = "" if v is None else str(v)

    session.run(
        f"""
        MERGE (e:`{label}` {{name: $name}})
        SET e += $props
        SET e.source_files =
            CASE
                WHEN $file_name IN coalesce(e.source_files, [])
                THEN e.source_files
                ELSE coalesce(e.source_files, []) + $file_name
            END
        """,
        {"name": name, "props": safe_props, "file_name": file_name},
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
        MATCH (a {{name: $from_name}})
        MATCH (b {{name: $to_name}})
        MERGE (a)-[:`{rel}`]->(b)
        """,
        {"from_name": from_name, "to_name": to_name},
    )


def build_graph():
    """전체 그래프 구축
    1단계: solar-pro 로 각 문서에서 개체·관계 동적 추출 (요약 없이 원문 직접 입력)
    2단계: solar-pro 로 전체 노드에서 유사 노드 통합 (1차/2차 모두 배치 처리)
    3단계: Neo4j 저장 (Document 노드 + MENTIONS 관계 포함)

    데이터 소스: manual_files + notices 첨부파일 + notices 공지 본문(content)
    """
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")
    notices_path = os.path.join(PARSED_DIR, "notices_parsed.json")

    files_to_process = []

    # 매뉴얼 파일들
    if os.path.exists(manual_path):
        with open(manual_path, encoding="utf-8") as f:
            for mf in json.load(f):
                text = str(mf.get("parsed_text", "")).strip()
                if text:
                    files_to_process.append({
                        "file_name": mf.get("file_name", "unknown"),
                        "parsed_text": text,
                    })
    else:
        print("[경고] manual_parsed.json 없음 — 매뉴얼 스킵")

    # 공지 첨부파일 + 공지 본문
    if os.path.exists(notices_path):
        with open(notices_path, encoding="utf-8") as f:
            for notice in json.load(f):
                title = (notice.get("title") or "").strip()
                short_title = title[:40]

                # 첨부파일
                for att in notice.get("attachments", []):
                    text = str(att.get("parsed_text", "")).strip()
                    if not text:
                        continue
                    files_to_process.append({
                        "file_name": f"[공지] {short_title} | {att.get('name', '')}",
                        "parsed_text": text,
                    })

                # 공지 본문(content) — 첨부파일 파싱 실패 시에도 정보 손실 없도록
                content = (notice.get("content") or "").strip()
                if content:
                    files_to_process.append({
                        "file_name": f"[공지본문] {short_title}",
                        "parsed_text": content,
                    })
    else:
        print("[경고] notices_parsed.json 없음 — 공지 스킵")

    if not files_to_process:
        print("[오류] 처리할 파일 없음")
        return

    driver = get_driver()

    try:
        clear_db(driver)

        print(f"[시작] Graph DB 구축 ({len(files_to_process)}개 파일) — solar-pro3 동적 추출 모드")

        all_entity_names = []
        extraction_results = []

        # ── 1단계: 각 문서를 청크로 분할 후 청크별 추출 → 문서 단위로 병합 ──
        for i, mf in enumerate(files_to_process):
            file_name = mf.get("file_name", f"unknown_{i}")
            parsed_text = str(mf.get("parsed_text", "")).strip()

            if not parsed_text:
                print(f"[{i+1}/{len(files_to_process)}] {file_name[:40]} → 텍스트 없음 스킵")
                continue

            chunks = split_into_chunks(parsed_text)
            print(f"\n[{i+1}/{len(files_to_process)}] {file_name[:60]} ({len(parsed_text)}자 → {len(chunks)}청크)")

            chunk_results = []
            for c_idx, chunk_text in enumerate(chunks):
                print(f"  청크 {c_idx+1}/{len(chunks)} ({len(chunk_text)}자) 추출 중...")
                result = extract_entities_and_relations(file_name, chunk_text)
                chunk_results.append(result)

            # 청크별 결과를 문서 단위로 병합 (중복 엔티티·관계 제거)
            extracted = merge_chunk_extractions(chunk_results)

            entities_count = len(extracted.get("entities", []))
            relations_count = len(extracted.get("relations", []))
            print(f"  병합 결과 — 개체: {entities_count}개, 관계: {relations_count}개")

            extraction_results.append((file_name, extracted))

            for ent in extracted.get("entities", []):
                name = normalize_node_name(str(ent.get("name", "")))
                if name and name != "Unknown" and not should_skip_entity_name(name):
                    all_entity_names.append(name)

        # ── 2단계: 유사 노드 통합 (1차/2차 모두 CONSOLIDATE_BATCH 단위 배치) ──
        unique_names = list(dict.fromkeys(all_entity_names))
        print(f"\n[노드 통합] 고유 개체명 {len(unique_names)}개 → solar-pro 유사도 분석 중...")

        name_mapping: dict[str, str] = {}

        # 1차 배치 통합
        for start in range(0, len(unique_names), CONSOLIDATE_BATCH):
            batch_names = unique_names[start:start + CONSOLIDATE_BATCH]
            batch_map = consolidate_nodes(batch_names)
            name_mapping.update(batch_map)
            print(f"  1차 배치 {start // CONSOLIDATE_BATCH + 1}: {len(batch_names)}개 처리 완료")

        # 2차 교차 통합 — canonical name도 배치로 처리해 프롬프트 과부하 방지
        canonical_names = list(dict.fromkeys(name_mapping.values()))
        print(f"\n[교차 통합] 1차 대표 이름 {len(canonical_names)}개 → 최종 통합 중...")

        if len(canonical_names) > 1:
            final_map: dict[str, str] = {}
            for start in range(0, len(canonical_names), CONSOLIDATE_BATCH):
                batch_names = canonical_names[start:start + CONSOLIDATE_BATCH]
                batch_map = consolidate_nodes(batch_names)
                final_map.update(batch_map)
                print(f"  2차 배치 {start // CONSOLIDATE_BATCH + 1}: {len(batch_names)}개 처리 완료")

            name_mapping = {
                orig: final_map.get(canonical, canonical)
                for orig, canonical in name_mapping.items()
            }
            print(f"  최종 대표 이름 {len(set(name_mapping.values()))}개로 통합 완료")

        # ── 3단계: Neo4j 저장 ──
        print("\n[저장] Neo4j에 노드·관계 저장 중...")

        with driver.session() as session:
            for file_name, extracted in extraction_results:
                create_document_node(session, file_name)

                valid_names = set()

                for ent in extracted.get("entities", []):
                    raw_name = normalize_node_name(str(ent.get("name", "")))

                    if not raw_name or raw_name == "Unknown" or should_skip_entity_name(raw_name):
                        continue

                    canonical = normalize_node_name(name_mapping.get(raw_name, raw_name))

                    if not canonical or canonical == "Unknown" or should_skip_entity_name(canonical):
                        continue

                    created = upsert_entity(
                        session,
                        canonical,
                        ent.get("type", "Entity"),
                        ent.get("properties", {}),
                        file_name,
                    )

                    if created:
                        valid_names.add(canonical)

                        session.run(
                            """
                            MATCH (d:Document {file_name: $file_name})
                            MATCH (e {name: $name})
                            MERGE (d)-[:MENTIONS]->(e)
                            """,
                            {"file_name": file_name, "name": canonical},
                        )

                for rel in extracted.get("relations", []):
                    from_raw = normalize_node_name(str(rel.get("from", "")))
                    to_raw = normalize_node_name(str(rel.get("to", "")))
                    rel_type = rel.get("type", "RELATED_TO")

                    if (
                        not from_raw or not to_raw
                        or from_raw == "Unknown" or to_raw == "Unknown"
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
                   OR ANY(f IN coalesce(n.source_files, []) WHERE f CONTAINS $keyword)
                   OR coalesce(m.name, '') CONTAINS $keyword
                   OR coalesce(m.title, '') CONTAINS $keyword
                   OR coalesce(m.file_name, '') CONTAINS $keyword
                   OR ANY(f IN coalesce(m.source_files, []) WHERE f CONTAINS $keyword)
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
                MATCH (n)
                WHERE n.name IS NOT NULL AND NOT n:Document
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
