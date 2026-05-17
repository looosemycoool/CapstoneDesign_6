# 03_graph_db.py 재설계 검토

작성일: 2026-05-13

---

## 1. 현재 설계의 문제점

### 1-1. LLM 통합(`run_consolidation`) — 가장 큰 병목

```
청크 N개 → 엔티티 E개 추출
→ 1차 배치 (E ÷ 80) 회 LLM 호출
→ 2차 교차 배치 (canonical 수 ÷ 80) 회 LLM 호출
→ 다음 문서 처리 시 기존 엔티티까지 포함해서 반복
```

문서 1개당 추출 LLM 호출 수 외에 통합용 LLM 호출이 수 배로 발생한다.
`existing_entity_names`를 문서 처리마다 누적하면서 통합 대상이 기하급수적으로 증가하는 구조다.

**실제 예시**: 10개 문서, 문서당 청크 5개, 청크당 엔티티 30개 추출 시
- 추출 LLM 호출: 50회
- 통합 LLM 호출: 문서 1 → 약 3회, 문서 10 → 기존 1000+개 포함해 약 20회+
- 총합: 추출 50회 + 통합 150+회 → **실제 추출의 3배 이상**

### 1-2. 엔티티·관계 수 제한 없음

`extract_from_chunk()` 프롬프트에 최대 개수 명시가 없다.
`entity_extractor.py`는 프롬프트에 "엔티티 최대 15개, 관계 최대 20개" 명시 + 후처리 슬라이싱 모두 있는데,
`03_graph_db.py`에는 어느 쪽도 없어서 청크당 40–60개 엔티티가 추출되는 경우가 생긴다.

### 1-3. `response_format` 미사용 → 파싱 불안정

`extract_from_chunk()`는 `response_format={"type": "json_object"}`를 쓰지 않아서
LLM이 마크다운 코드블록(```json)으로 감싸거나 설명 문장을 앞에 붙이면 `safe_json_load()`로 파싱한다.
이 파싱 로직이 실패하면 청크 전체 데이터가 손실된다.

### 1-4. 추출과 저장이 하나의 함수에 혼재

`process_document()`가 청킹 → 추출 → 통합 → 저장을 모두 처리한다.
`entity_extractor.py`와 `neo4j_indexer.py`처럼 레이어를 분리하면 테스트·재사용·독립 개선이 가능한데,
지금은 중간 결과를 확인하거나 저장 로직만 바꾸기가 어렵다.

### 1-5. CO_OCCURS 관계 없음

`neo4j_indexer.py`에는 같은 청크에서 함께 등장한 엔티티 쌍을 자동으로 `CO_OCCURS`로 연결하는 로직이 있다.
`03_graph_db.py`에는 없어서 통계적 공출현 신호를 그래프에서 활용할 수 없다.

### 1-6. 동기 방식 + 순차 청크 처리

`openai.OpenAI` (동기 클라이언트)를 사용하고 청크를 순차로 처리한다.
`entity_extractor.py`는 `AsyncOpenAI`를 사용한다. 파이프라인 수준에서 asyncio를 쓰면 청크 처리를 병렬화할 수 있는데 지금은 불가능하다.

### 1-7. 도메인 스키마 없는 범용 프롬프트

17종 관계 타입 목록은 있으나 경북대 공지 도메인에 특화된 노드 타입 정의와 추출 가이드가 없다.
결과적으로 "GraduationRequirement", "Credit" 같은 영문 캐멀케이스 엔티티가 여전히 추출된다.

---

## 2. 재설계 방향

### 핵심 원칙

> 청크당 LLM 호출 1회, 통합 LLM 호출 0회, 엔티티 노이즈는 프롬프트 + 규칙으로 제어

### 우선순위별 변경

| # | 변경 | 현재 | 목표 |
|---|------|------|------|
| 1 | 추출-저장 레이어 분리 | 혼재 | `_extract` / `_index` 섹션 분리 |
| 2 | `response_format` 적용 | 미사용 | `json_object` 강제 |
| 3 | 엔티티·관계 상한 | 무제한 | 프롬프트 15/20개 + 후처리 슬라이싱 |
| 4 | LLM 통합 제거 | 2단계 배치 통합 | `ALIAS_MAP` 규칙 기반으로 교체 |
| 5 | CO_OCCURS 추가 | 없음 | 청크 저장 후 자동 생성 |
| 6 | 경북대 도메인 스키마 | 범용 17종 관계 | 도메인 특화 노드+관계 스키마 |

---

## 3. 경북대 공지 도메인 스키마

### 3-1. 노드 타입 (8종)

| 레이블 | 설명 | 추출 단서 |
|--------|------|-----------|
| `Department` | 학부·학과·단과대 | "∼학부", "∼대학", "∼학과" |
| `Major` | 전공·이중전공 | "복수전공", "부전공", "연계전공", "학생설계전공", "다전공" |
| `Program` | 제도·프로그램 | "조기졸업", "장학금", "교환학생", "봉사" |
| `Course` | 강좌·교과목 | "전공필수", "교양", "졸업논문", "수강" |
| `Requirement` | 이수·졸업 요건 | "∼이상", "∼학점", "GPA", "평점", "이수" |
| `Period` | 기간·일정 | "∼까지", "∼기간", "신청 기간", "년 월" |
| `Organization` | 외부 기관·기업 | "∼기관", "∼협회", "∼회사" |
| `Person` | 교수·직원 | "교수", "직원", "담당자" |

### 3-2. 관계 타입 (10종 → 기존 17종에서 핵심만)

| 관계 | 방향 | 의미 |
|------|------|------|
| `REQUIRES` | Requirement → ∗ | A를 이수/충족해야 B 가능 |
| `BELONGS_TO` | ∗ → Department | 소속 |
| `OFFERS` | Department → Course/Program | 개설·제공 |
| `APPLIES_TO` | Program/Requirement → Major/Department | 적용 대상 |
| `HAS_DEADLINE` | Program/Requirement → Period | 기한 |
| `EXCLUDES` | ∗ → ∗ | 제외·불인정 |
| `SUBSTITUTES_FOR` | Course/Major → Course/Major | 대체 인정 |
| `PART_OF` | ∗ → Program | 구성 요소 |
| `PROVIDES` | Organization → Program | 후원·제공 |
| `RELATED_TO` | ∗ → ∗ | 기타 관계 (fallback) |

### 3-3. 관계 자동 생성 (규칙 기반)

- `CO_OCCURS`: 같은 청크에서 함께 언급된 엔티티 쌍 (통계적 공출현, `count` 속성 증가)

---

## 4. 제안 파일 구조

```
03_graph_db.py
│
├── ── 설정 / 상수 ──────────────────────────────────────────
│   ├── 환경변수 로드 (NEO4J_*, UPSTAGE_API_KEY)
│   ├── ALIAS_MAP          # 도메인 동의어 사전 (확장)
│   └── DOMAIN_SCHEMA      # 경북대 노드/관계 스키마 (프롬프트용)
│
├── ── 유틸 ──────────────────────────────────────────────────
│   ├── normalize_name()
│   ├── make_doc_key() / make_chunk_id()
│   ├── content_hash()
│   └── load_state() / save_state()
│
├── ── 문서 로드 / 청킹 ──────────────────────────────────────
│   ├── load_documents()
│   └── split_document()
│
├── [추출 레이어] ────────────────────────────────────────────
│   ├── KNU_EXTRACTION_PROMPT   # 도메인 스키마 포함, 15/20개 상한
│   └── extract_from_chunk(file_name, chunk_text) -> dict
│       ├── response_format={"type": "json_object"}
│       ├── 후처리: entities[:15], relations[:20]
│       └── normalize_name 적용 (LLM 통합 없음)
│
├── [저장 레이어 (Neo4j)] ────────────────────────────────────
│   ├── ensure_constraints()    # 제약·인덱스 생성 (1회)
│   ├── write_document()
│   ├── write_chunk()
│   ├── write_entities_and_mentions()
│   ├── write_relations()
│   └── write_co_occurs()       # 신규: CO_OCCURS 자동 생성
│
├── ── 문서 단위 처리 ────────────────────────────────────────
│   └── process_document(driver, doc) -> list[str]
│       ├── split_document()
│       ├── for chunk: extract_from_chunk()   # 순차 (추후 async 전환 가능)
│       └── with session: write_* 일괄 저장
│
└── ── 메인 ─────────────────────────────────────────────────
    ├── build_graph(rebuild=False)
    └── check_graph()
```

---

## 5. 핵심 변경 상세

### 5-1. LLM 통합 제거

**현재**
```python
# 문서마다 모든 엔티티 이름을 LLM에 넘겨서 통합
name_mapping = run_consolidation(combined)  # 수십 번 LLM 호출
```

**변경 후**
```python
# ALIAS_MAP 확장 + normalize_name()으로 규칙 기반 처리만
# Neo4j MERGE가 이미 exact-match 중복을 방지함
canonical = normalize_name(raw_name)  # LLM 없이 즉시 처리
```

ALIAS_MAP에 도메인 주요 동의어를 미리 등록한다:
```python
ALIAS_MAP = {
    # 기존
    "글솝": "글로벌소프트웨어융합전공",
    "IT대학": "경북대학교IT대학",
    # 추가
    "다전공": "다전공프로그램",
    "복전": "복수전공",
    "부전": "부전공",
    "조기졸": "조기졸업",
    "수강신청": "수강신청기간",
    "장학": "장학금",
    "GPA": "평점평균",
    "학점": "이수학점",
}
```

**트레이드오프**: 사전에 없는 새 동의어는 별도 노드로 저장된다.
이는 노이즈 증가보다 LLM 통합 비용 절감이 훨씬 크다고 판단.
향후 데이터가 쌓이면 Neo4j에서 `ALIAS` 엣지 쿼리로 사후 통합 가능.

### 5-2. 추출 프롬프트 — 경북대 도메인 스키마 주입

```
[경북대 공지 도메인 스키마]
노드 타입: Department, Major, Program, Course, Requirement, Period, Organization, Person
관계 타입: REQUIRES, BELONGS_TO, OFFERS, APPLIES_TO, HAS_DEADLINE,
           EXCLUDES, SUBSTITUTES_FOR, PART_OF, PROVIDES, RELATED_TO

규칙:
- 노드 이름은 반드시 한국어 (영문 캐멀케이스 금지)
- 엔티티 최대 15개, 관계 최대 20개
- 추론 금지 — 텍스트에 명시된 것만 추출
- 날짜·기간은 Period 노드에 저장
- 수치 요건(평점, 학점)은 Requirement 노드 properties에 저장
```

### 5-3. `response_format` + 후처리 슬라이싱

```python
response = upstage_client.chat.completions.create(
    model=EXTRACT_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    response_format={"type": "json_object"},  # 추가
    max_tokens=2048,  # 4096 → 2048 (상한 줄여서 노이즈 억제)
)
data = json.loads(response.choices[0].message.content or "{}")

entities = data.get("entities", [])[:15]   # 슬라이싱
relations = data.get("relations", [])[:20]  # 슬라이싱
```

`safe_json_load()` 함수는 제거한다.

### 5-4. CO_OCCURS 자동 생성

`neo4j_indexer.py`의 패턴을 그대로 채용:

```python
def write_co_occurs(session, chunk_id: str):
    """같은 청크에서 함께 등장한 엔티티 쌍에 CO_OCCURS 관계 생성."""
    session.run(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e1:Entity)
        MATCH (c)-[:MENTIONS]->(e2:Entity)
        WHERE e1.name < e2.name
        MERGE (e1)-[r:CO_OCCURS]->(e2)
        ON CREATE SET r.count = 1
        ON MATCH SET r.count = r.count + 1
        """,
        chunk_id=chunk_id,
    )
```

CO_OCCURS는 LLM 추출 없이 저장 단계에서 자동 생성된다.
04_hybrid_rag.py의 `search_by_entities()`가 이미 `CO_OCCURS`를 2-hop 탐색에 활용하도록 설계되어 있으므로 바로 연동 가능.

### 5-5. ensure_constraints() 추가

빌드 시작 시 1회만 실행:

```python
def ensure_constraints(driver):
    with driver.session() as s:
        s.run("CREATE CONSTRAINT doc_key IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_key IS UNIQUE")
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
        s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
```

제약이 있으면 MERGE 성능이 O(log n)으로 개선된다.

---

## 6. 예상 성능 개선

| 항목 | 현재 | 변경 후 | 절감 |
|------|------|---------|------|
| 청크당 LLM 호출 | 1회 | 1회 | — |
| 문서당 통합 LLM 호출 | 2–20회+ | **0회** | ∞ |
| 응답 파싱 실패율 | 높음 (마크다운 파싱) | 낮음 (json_object) | ↓ |
| 청크당 평균 엔티티 | 무제한 (30–60개) | 최대 15개 | ↓50–75% |
| 그래프 노이즈 | 높음 | 낮음 | — |
| Neo4j MERGE 성능 | 인덱스 없음 | 유니크 제약 있음 | ↑ |

---

## 7. 04_hybrid_rag.py 호환성

현재 `04_hybrid_rag.py`는 다음 구조를 가정한다:

```
(Document)-[:HAS_CHUNK]->(Chunk)    ← 현재 03_graph_db.py 사용
(Chunk)-[:MENTIONS]->(Entity)
(Entity)-[RELATED_TO|CO_OCCURS]->(Entity)
```

`neo4j_indexer.py`는 `(Document)-[:CONTAINS]->(Chunk)` (관계명 다름)를 사용하므로
**03_graph_db.py는 기존 `HAS_CHUNK` 관계명을 유지**하여 04와의 호환성을 깨지 않는다.

CO_OCCURS 추가는 04_hybrid_rag.py가 이미 지원하므로 별도 수정 불필요.

---

## 8. 미결 사항 (구현 전 확인 필요)

1. **ALIAS_MAP 범위**: 어느 수준까지 동의어를 등록할지 결정 필요.
   현재 ALIAS_MAP 8개 → 도메인 주요 용어 기준 20–30개로 확장 예정.

2. **청크 크기 조정 여부**: `GRAPH_CHUNK_SIZE=1500`은 유지하되,
   엔티티 상한 15개와 맞춰 청크당 정보량이 적절한지 실제 문서 기준으로 검증 필요.

3. **기존 그래프 데이터 마이그레이션**: 재설계 후 `--rebuild` 실행 시 기존 노드는 전부 삭제된다.
   CO_OCCURS 관계가 없는 기존 그래프를 재활용할지, 처음부터 재구축할지 결정 필요.
   → **재구축 권장** (통합 제거로 노드 명칭이 달라질 수 있음).

4. **`solar-pro3` 모델의 `response_format` 지원 여부**: Upstage 공식 문서에서
   `json_object` 지원을 확인 필요. `entity_extractor.py`에서 이미 사용 중이므로 지원되는 것으로 추정.

---

## 9. 구현 순서 (제안)

```
Step 1: 상수/유틸 섹션 재작성 (ALIAS_MAP 확장, DOMAIN_SCHEMA 정의)
Step 2: extract_from_chunk() 교체 (response_format + 상한 + 도메인 프롬프트)
Step 3: LLM 통합 함수(consolidate_names, run_consolidation) 제거
Step 4: process_document() 단순화 (통합 루프 제거)
Step 5: ensure_constraints() + write_co_occurs() 추가
Step 6: --rebuild 실행 후 그래프 품질 검증 (check_graph())
```
