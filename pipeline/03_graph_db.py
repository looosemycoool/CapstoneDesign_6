"""Graph DB 구축 — Document / Chunk / Entity 3계층 구조.

설계 원칙:
  - Chunk 노드의 chunk_id = ChromaDB chunk_id (02_vector_db.py 와 동일 규칙)
  - HybridRAG 검색: chunk_id 브리지로 VectorDB ↔ GraphDB 연결
  - 증분 처리: data/graph_state.json 으로 처리된 파일 추적
  - LLM 통합 없음: ALIAS_MAP 규칙 기반 정규화 + Neo4j MERGE 중복 방지

노드:
  Document  {doc_key, file_name, source_type, date, notice_title, content_hash}
  Chunk     {chunk_id, doc_key, chunk_index, text_preview}
  Entity    {name, type}

엣지:
  (Document)-[:HAS_CHUNK]->(Chunk)
  (Chunk)-[:MENTIONS]->(Entity)
  (Entity)-[:REL_TYPE {evidence_chunk_id, source_doc_key}]->(Entity)
  (Entity)-[:CO_OCCURS {count}]->(Entity)

실행:
  python pipeline/03_graph_db.py            # 증분 처리 (새 파일만)
  python pipeline/03_graph_db.py --rebuild  # 전체 재구축
"""
import os
import re
import sys
import json
import hashlib
import argparse
import traceback
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
STATE_PATH = os.path.join(BASE_DIR, "data", "graph_state.json")

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
EXTRACT_MODEL = "solar-pro3"

# 청킹 설정 — 02_vector_db.py 와 동일 규칙 (chunk_id 브리지 정합성 필수)
GRAPH_CHUNK_SIZE = 800
GRAPH_CHUNK_OVERLAP = 150

# ── 추출 상한 ────────────────────────────────────────────────
MAX_ENTITIES = 15
MAX_RELATIONS = 20


# ══════════════════════════════════════════════════════════════
# Step 1: 상수 / 유틸
# ══════════════════════════════════════════════════════════════

# 경북대 도메인 동의어 사전 — LLM 통합 없이 규칙 기반 정규화
ALIAS_MAP: dict[str, str] = {
    # 전공 계열
    "글솝":               "글로벌소프트웨어융합전공",
    "글로벌소프트웨어":    "글로벌소프트웨어융합전공",
    "글로벌SW":            "글로벌소프트웨어융합전공",
    "글로벌 소프트웨어":   "글로벌소프트웨어융합전공",
    "다전공":             "다전공프로그램",
    "복전":               "복수전공",
    "부전":               "부전공프로그램",
    "부전공":             "부전공프로그램",
    "연계전공":            "연계전공프로그램",
    "학생설계전공":        "학생설계전공프로그램",
    # 졸업
    "조기 졸업":           "조기졸업제도",
    "조기졸업":            "조기졸업제도",
    "조기졸":              "조기졸업제도",
    # 기관
    "IT대학":              "경북대학교IT대학",
    "컴퓨터학부":          "경북대학교컴퓨터학부",
    "컴학부":              "경북대학교컴퓨터학부",
    "경북대":              "경북대학교",
    "경대":                "경북대학교",
    # 학점·성적
    "GPA":                "평점평균",
    "학점":               "이수학점",
    "졸업학점":            "졸업이수학점",
    # 장학금
    "장학":               "장학금",
    "국가장학금":          "국가장학금제도",
    # 수강
    "수강신청":            "수강신청기간",
}

_NULL_NAMES = {"none", "null", "n/a", "nan", "undefined", "unknown", ""}
SKIP_NAMES = {"홍길동"}


def normalize_name(text) -> str:
    if not isinstance(text, (str, int, float)) or text is None:
        return "Unknown"
    text = str(text).strip()
    if text.lower() in _NULL_NAMES:
        return "Unknown"
    text = re.sub(r"\s+", " ", text)
    if text in ALIAS_MAP:
        return ALIAS_MAP[text]
    no_paren = re.sub(r"\(.*?\)", "", text).strip()
    no_paren = re.sub(r"\s+", " ", no_paren)
    if no_paren in ALIAS_MAP:
        return ALIAS_MAP[no_paren]
    return text or "Unknown"


def is_valid_name(name: str) -> bool:
    return bool(name) and name != "Unknown" and name not in SKIP_NAMES


def sanitize_rel_type(rel_type: str) -> str:
    cleaned = re.sub(r"[^\w]", "_", (rel_type or "RELATED_TO").strip().upper(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "RELATED_TO"


def make_doc_key(source_type: str, file_name: str, notice_num: str = "") -> str:
    if source_type == "manual":
        return f"manual::{file_name}"
    elif source_type == "notice":
        return f"notice::{notice_num}::{file_name}"
    elif source_type == "notice_content":
        return f"notice_content::{notice_num}"
    return f"{source_type}::{file_name}"


def make_chunk_id(doc_key: str, chunk_index: int) -> str:
    return f"{doc_key}::chunk{chunk_index}"


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# 문서 로드 / 청킹
# ══════════════════════════════════════════════════════════════

def load_documents() -> list[dict]:
    docs = []

    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")
    if os.path.exists(manual_path):
        with open(manual_path, encoding="utf-8") as f:
            for m in json.load(f):
                text = (m.get("parsed_text") or "").strip()
                if not text:
                    continue
                file_name = m.get("file_name", "unknown")
                docs.append({
                    "doc_key":      make_doc_key("manual", file_name),
                    "file_name":    file_name,
                    "source_type":  "manual",
                    "notice_title": "",
                    "notice_num":   "",
                    "date":         "",
                    "parsed_text":  text,
                })

    notices_path = os.path.join(PARSED_DIR, "notices_parsed.json")
    if os.path.exists(notices_path):
        with open(notices_path, encoding="utf-8") as f:
            seen_nums: set[str] = set()
            for n in json.load(f):
                num = str(n.get("num") or "unknown")
                if num in seen_nums:
                    continue
                seen_nums.add(num)
                title = (n.get("title") or "").strip()
                date  = (n.get("date")  or "").strip()

                for a in n.get("attachments", []):
                    text = (a.get("parsed_text") or "").strip()
                    if not text:
                        continue
                    att_name = a.get("name", "unknown")
                    docs.append({
                        "doc_key":      make_doc_key("notice", att_name, num),
                        "file_name":    att_name,
                        "source_type":  "notice",
                        "notice_title": title,
                        "notice_num":   num,
                        "date":         date,
                        "parsed_text":  text,
                    })

                content = (n.get("content") or "").strip()
                if content:
                    docs.append({
                        "doc_key":      make_doc_key("notice_content", "", num),
                        "file_name":    f"notice_{num}",
                        "source_type":  "notice_content",
                        "notice_title": title,
                        "notice_num":   num,
                        "date":         date,
                        "parsed_text":  content,
                    })

    return docs


_splitter = RecursiveCharacterTextSplitter(
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


def split_document(doc: dict) -> list[dict]:
    text    = doc["parsed_text"]
    doc_key = doc["doc_key"]
    chunks  = _splitter.split_text(text) or [text]
    return [
        {
            "chunk_id":    make_chunk_id(doc_key, i),
            "chunk_index": i,
            "text":        chunk_text,
            "text_preview": chunk_text[:100],
        }
        for i, chunk_text in enumerate(chunks)
    ]


# ══════════════════════════════════════════════════════════════
# Step 2: 추출 레이어 — 경북대 도메인 스키마 + response_format
# ══════════════════════════════════════════════════════════════

# 경북대 공지 도메인 스키마 (프롬프트 주입용)
_KNU_DOMAIN_SCHEMA = """[경북대 공지 도메인 스키마]

노드 타입 (이 8종만 사용):
  Department   : 학부·학과·단과대  (예: 경북대학교컴퓨터학부, IT대학)
  Major        : 전공·이중전공     (예: 복수전공, 부전공프로그램, 글로벌소프트웨어융합전공)
  Program      : 제도·프로그램     (예: 조기졸업제도, 장학금, 해외교환학생)
  Course       : 강좌·교과목       (예: 전공필수, 교양선택, 졸업논문)
  Requirement  : 이수·졸업 요건   (예: 졸업이수학점, 평점평균, 이수 조건)
  Period       : 기간·일정         (예: 수강신청기간, 장학금신청기간, 2024년 1학기)
  Organization : 외부 기관·기업   (예: 교육부, 한국장학재단)
  Person       : 교수·직원         (예: 담당교수, 학생처장)

관계 타입 (이 12종만 사용):
  REQUIRES       : A를 이수/충족해야 B 가능
  HAS_CONDITION  : 조건·자격 (예: 복학생은 신청 가능)
  HAS_EXCEPTION  : 예외·면제 (예: 외국인 유학생은 면제)
  BELONGS_TO     : 소속 (학과 → 단과대)
  OFFERS         : 개설·제공 (학부 → 강좌/프로그램)
  APPLIES_TO     : 적용 대상 (요건 → 전공/학과)
  HAS_DEADLINE   : 기한 (프로그램/요건 → 기간)
  EXCLUDES       : 제외·불인정
  SUBSTITUTES_FOR: 대체 인정
  PART_OF        : 구성 요소
  PROVIDES       : 후원·제공 (기관 → 프로그램)
  RELATED_TO     : 기타 관계 (위 11종에 맞지 않을 때만)"""

_EXTRACTION_PROMPT = """\
당신은 경북대학교 행정 문서에서 지식 그래프를 구축하는 전문가입니다.
아래 문서 청크에서 노드(개체)와 관계를 추출하세요.

{domain_schema}

파일명: {file_name}
---
{chunk_text}
---

## 추출 규칙
1. 노드 name 은 반드시 한국어 (영문 캐멀케이스 절대 금지)
2. type 은 위 8종 노드 타입 중 하나 (해당 없으면 Requirement 로 fallback)
3. 날짜·기간은 Period 노드로 생성
4. 텍스트에 명시된 것만 추출 — 추론 금지
5. 예시용 이름(홍길동), 빈 name 금지
6. relation 의 from·to 는 반드시 위 entities 의 name 중 하나
7. 엔티티 최대 {max_entities}개, 관계 최대 {max_relations}개

반드시 아래 JSON 형식으로만 응답하세요.

{{
  "entities": [
    {{"name": "개체명", "type": "노드타입"}}
  ],
  "relations": [
    {{"from": "개체명1", "to": "개체명2", "type": "관계타입"}}
  ]
}}"""


def _recover_partial_json(content: str) -> dict:
    """max_tokens 초과로 잘린 JSON에서 완전한 객체만 추출.

    중괄호 depth 추적으로 완전히 닫힌 {} 블록만 파싱하므로
    잘린 마지막 객체는 버리고 앞부분 완성된 것만 반환한다.
    """
    def extract_objects(text: str, section: str) -> list[dict]:
        start = text.find(f'"{section}"')
        if start == -1:
            return []
        arr_start = text.find('[', start)
        if arr_start == -1:
            return []
        objects = []
        depth = 0
        obj_start = None
        for i in range(arr_start, len(text)):
            ch = text[i]
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        objects.append(json.loads(text[obj_start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
            elif ch == ']' and depth == 0:
                break
        return objects

    return {
        "entities":  extract_objects(content, "entities"),
        "relations": extract_objects(content, "relations"),
    }


def extract_from_chunk(file_name: str, chunk_text: str) -> dict:
    """청크 텍스트에서 엔티티·관계 추출 (LLM 1회 호출)."""
    prompt = _EXTRACTION_PROMPT.format(
        domain_schema=_KNU_DOMAIN_SCHEMA,
        file_name=file_name,
        chunk_text=chunk_text[:3000],
        max_entities=MAX_ENTITIES,
        max_relations=MAX_RELATIONS,
    )
    try:
        resp = upstage_client.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        truncated = choice.finish_reason == "length"

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            if truncated:
                data = _recover_partial_json(content)
                e_cnt = len(data.get("entities", []))
                r_cnt = len(data.get("relations", []))
                print(f"    [부분 복구] 잘린 응답에서 엔티티 {e_cnt}개, 관계 {r_cnt}개 복구")
            else:
                raise

        if truncated and data.get("entities"):
            print(f"    [경고] max_tokens 초과 — {len(data.get('entities', []))}개 엔티티까지 저장")
        if not isinstance(data, dict):
            return {"entities": [], "relations": []}

        raw_entities  = data.get("entities",  []) if isinstance(data.get("entities"),  list) else []
        raw_relations = data.get("relations", []) if isinstance(data.get("relations"), list) else []

        # 상한 적용 + 필터링
        valid_entities: list[dict] = []
        entity_names:   set[str]   = set()
        for e in raw_entities[:MAX_ENTITIES]:
            if not isinstance(e, dict):
                continue
            name = normalize_name(e.get("name"))
            if not is_valid_name(name):
                continue
            e["name"] = name
            e["properties"] = {}
            valid_entities.append(e)
            entity_names.add(name)

        valid_relations: list[dict] = []
        for r in raw_relations[:MAX_RELATIONS]:
            if not isinstance(r, dict):
                continue
            fr = normalize_name(r.get("from"))
            to = normalize_name(r.get("to"))
            if fr in entity_names and to in entity_names and fr != to:
                r["from"], r["to"] = fr, to
                valid_relations.append(r)

        return {"entities": valid_entities, "relations": valid_relations}

    except Exception as e:
        print(f"    [추출 오류] {type(e).__name__}: {e}")
        return {"entities": [], "relations": []}


# ══════════════════════════════════════════════════════════════
# Step 5: 저장 레이어 (Neo4j)
# ══════════════════════════════════════════════════════════════

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ensure_constraints(driver):
    """유니크 제약 + 인덱스 생성 (빌드 시작 시 1회). MERGE 성능 O(log n)."""
    constraints = [
        "CREATE CONSTRAINT doc_key_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_key IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    ]
    with driver.session() as s:
        for cypher in constraints:
            try:
                s.run(cypher)
            except Exception as ex:
                print(f"  [제약 생성] {ex}")


def clear_db(driver):
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("[초기화] 기존 그래프 삭제 완료")


def write_document(session, doc: dict):
    session.run(
        """
        MERGE (d:Document {doc_key: $doc_key})
        SET d.file_name     = $file_name,
            d.source_type   = $source_type,
            d.notice_title  = $notice_title,
            d.date          = $date,
            d.content_hash  = $content_hash
        """,
        {
            "doc_key":      doc["doc_key"],
            "file_name":    doc["file_name"],
            "source_type":  doc["source_type"],
            "notice_title": doc.get("notice_title", ""),
            "date":         doc.get("date", ""),
            "content_hash": content_hash(doc["parsed_text"]),
        },
    )


def write_chunk(session, doc_key: str, chunk: dict):
    session.run(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.doc_key      = $doc_key,
            c.chunk_index  = $chunk_index,
            c.text_preview = $text_preview
        WITH c
        MATCH (d:Document {doc_key: $doc_key})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        {
            "chunk_id":    chunk["chunk_id"],
            "doc_key":     doc_key,
            "chunk_index": chunk["chunk_index"],
            "text_preview": chunk["text_preview"],
        },
    )


def write_entity(session, name: str, etype: str, props: dict):
    safe_props = {k: v for k, v in props.items() if k not in {"name", "type"}}
    session.run(
        """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type
        ON MATCH SET  e.type = CASE WHEN e.type IS NULL THEN $type ELSE e.type END
        SET e += $props
        """,
        {"name": name, "type": etype or "Requirement", "props": safe_props},
    )


def write_mentions(session, chunk_id: str, entity_name: str):
    session.run(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MATCH (e:Entity {name: $name})
        MERGE (c)-[:MENTIONS]->(e)
        """,
        {"chunk_id": chunk_id, "name": entity_name},
    )


def write_relation(session, from_name: str, to_name: str, rel_type: str,
                   evidence_chunk_id: str, source_doc_key: str):
    rel = sanitize_rel_type(rel_type)
    session.run(
        f"""
        MATCH (a:Entity {{name: $from_name}})
        MATCH (b:Entity {{name: $to_name}})
        MERGE (a)-[r:`{rel}`]->(b)
        SET r.evidence_chunk_id = $evidence_chunk_id,
            r.source_doc_key    = $source_doc_key
        """,
        {
            "from_name":        from_name,
            "to_name":          to_name,
            "evidence_chunk_id": evidence_chunk_id,
            "source_doc_key":   source_doc_key,
        },
    )


def write_co_occurs(session, chunk_id: str):
    """같은 청크에 함께 등장한 엔티티 쌍에 CO_OCCURS 관계 생성 (count 누적)."""
    session.run(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e1:Entity)
        MATCH (c)-[:MENTIONS]->(e2:Entity)
        WHERE e1.name < e2.name
        MERGE (e1)-[r:CO_OCCURS]->(e2)
        ON CREATE SET r.count = 1
        ON MATCH SET  r.count = r.count + 1
        """,
        {"chunk_id": chunk_id},
    )


# ══════════════════════════════════════════════════════════════
# Step 4: 문서 단위 처리 (통합 루프 없음)
# ══════════════════════════════════════════════════════════════

def process_document(driver, doc: dict) -> int:
    """단일 문서: 청크 분할 → 추출 → Neo4j 저장. 저장된 엔티티 수 반환."""
    chunks = split_document(doc)
    print(f"  {len(chunks)}개 청크 처리 중...")

    with driver.session() as session:
        write_document(session, doc)

        total_entities = 0
        for chunk in chunks:
            result = extract_from_chunk(doc["file_name"], chunk["text"])
            e_cnt = len(result["entities"])
            r_cnt = len(result["relations"])
            print(f"    청크 {chunk['chunk_index']}: 개체 {e_cnt}개, 관계 {r_cnt}개")

            write_chunk(session, doc["doc_key"], chunk)

            # 엔티티 저장 + MENTIONS
            saved_names: set[str] = set()
            for e in result["entities"]:
                name = e["name"]
                write_entity(session, name, e.get("type", "Requirement"), e.get("properties", {}))
                write_mentions(session, chunk["chunk_id"], name)
                saved_names.add(name)

            # 명시적 관계 저장
            for r in result["relations"]:
                fr, to = r["from"], r["to"]
                if fr in saved_names and to in saved_names:
                    write_relation(session, fr, to, r["type"],
                                   chunk["chunk_id"], doc["doc_key"])

            # CO_OCCURS 자동 생성 (엔티티가 2개 이상일 때만)
            if len(saved_names) >= 2:
                write_co_occurs(session, chunk["chunk_id"])

            total_entities += e_cnt

    return total_entities


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def build_graph(rebuild: bool = False):
    docs  = load_documents()
    state = load_state()

    driver = get_driver()

    if rebuild:
        clear_db(driver)
        state = {}

    # 제약·인덱스 보장 (Step 5)
    ensure_constraints(driver)

    to_process = [
        d for d in docs
        if rebuild or state.get(d["doc_key"]) != content_hash(d["parsed_text"])
    ]

    skipped = len(docs) - len(to_process)
    print(f"[시작] 전체 {len(docs)}개 문서 — 처리 {len(to_process)}개, 스킵 {skipped}개")

    if not to_process:
        print("[완료] 새로운 문서 없음")
        driver.close()
        return

    try:
        for i, doc in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}] {doc['doc_key']}")
            try:
                entity_cnt = process_document(driver, doc)
                print(f"  → 엔티티 {entity_cnt}개 저장")
                state[doc["doc_key"]] = content_hash(doc["parsed_text"])
                save_state(state)
            except Exception as e:
                print(f"  [오류] {type(e).__name__}: {e}")
                print(traceback.format_exc())

        # 최종 통계
        with driver.session() as s:
            doc_cnt    = s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]
            chunk_cnt  = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            entity_cnt = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            rel_cnt    = s.run("MATCH ()-[r]->() WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS' RETURN count(r) AS c").single()["c"]
            co_cnt     = s.run("MATCH ()-[r:CO_OCCURS]->() RETURN count(r) AS c").single()["c"]
            sem_cnt    = rel_cnt - co_cnt

        print("\n[완료] Graph DB 구축 완료!")
        print(f"  Document:   {doc_cnt}개")
        print(f"  Chunk:      {chunk_cnt}개")
        print(f"  Entity:     {entity_cnt}개")
        print(f"  의미 관계:  {sem_cnt}개")
        print(f"  CO_OCCURS:  {co_cnt}개")
        print("\n  Neo4j 확인: http://localhost:7474")

    finally:
        driver.close()


def check_graph():
    """그래프 구조 검증 — 중복 Entity, 고아 Chunk, CO_OCCURS 분포 확인."""
    driver = get_driver()
    try:
        with driver.session() as s:
            # 중복 Entity
            dups = s.run("""
                MATCH (e:Entity)
                WITH e.name AS name, count(e) AS cnt
                WHERE cnt > 1
                RETURN name, cnt ORDER BY cnt DESC LIMIT 10
            """).data()
            if dups:
                print("[중복 Entity]")
                for r in dups:
                    print(f"  {r['name']}: {r['cnt']}개")
            else:
                print("[중복 Entity] 없음")

            # MENTIONS 없는 고아 Chunk
            orphan = s.run("""
                MATCH (c:Chunk)
                WHERE NOT (c)-[:MENTIONS]->()
                RETURN count(c) AS cnt
            """).single()["cnt"]
            print(f"[MENTIONS 없는 Chunk] {orphan}개")

            # CO_OCCURS 상위 10개
            top_co = s.run("""
                MATCH (e1:Entity)-[r:CO_OCCURS]->(e2:Entity)
                RETURN e1.name AS a, e2.name AS b, r.count AS cnt
                ORDER BY cnt DESC LIMIT 10
            """).data()
            if top_co:
                print("\n[CO_OCCURS 상위 10]")
                for r in top_co:
                    print(f"  {r['a']} ↔ {r['b']} (count={r['cnt']})")

            # 의미 관계 타입별 분포
            rel_dist = s.run("""
                MATCH ()-[r]->()
                WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS' AND type(r) <> 'CO_OCCURS'
                RETURN type(r) AS rel, count(r) AS cnt
                ORDER BY cnt DESC LIMIT 15
            """).data()
            if rel_dist:
                print("\n[의미 관계 분포]")
                for r in rel_dist:
                    print(f"  {r['rel']}: {r['cnt']}개")

            # 샘플 경로
            print("\n[샘플 경로] Chunk → Entity → Entity")
            rows = s.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e1:Entity)-[r]->(e2:Entity)
                WHERE type(r) <> 'CO_OCCURS'
                RETURN c.chunk_id AS chunk, e1.name AS from, type(r) AS rel, e2.name AS to
                LIMIT 5
            """).data()
            for row in rows:
                print(f"  [{row['chunk'][:50]}]")
                print(f"    {row['from']} --[{row['rel']}]--> {row['to']}")

    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="기존 그래프 삭제 후 전체 재구축")
    args = parser.parse_args()

    build_graph(rebuild=args.rebuild)
    check_graph()
