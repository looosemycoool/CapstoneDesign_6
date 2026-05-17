# KNU CS 학과 공지사항 Hybrid RAG 챗봇

경북대학교 컴퓨터학부 공지사항·규정을 기반으로 학생 질문에 답변하는 **Hybrid RAG (Retrieval-Augmented Generation)** 시스템입니다.

벡터 검색(ChromaDB)과 지식 그래프(Neo4j)를 결합하여, 단순 키워드 매칭으로는 어려운 조건 해석·다문서 결합형 질문까지 처리합니다.

---

## 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [전처리 파이프라인](#전처리-파이프라인)
3. [Hybrid RAG 엔진](#hybrid-rag-엔진)
4. [평가 시스템](#평가-시스템)
5. [설치 및 환경 구성](#설치-및-환경-구성)
6. [실행 방법](#실행-방법)
7. [환경 변수 레퍼런스](#환경-변수-레퍼런스)
8. [디렉토리 구조](#디렉토리-구조)
9. [주요 설계 결정 및 Ablation 결과](#주요-설계-결정-및-ablation-결과)
10. [변경 이력](#변경-이력)

---

## 시스템 아키텍처

```
[공지 게시판 크롤링]
       │
       ▼
[문서 파싱] (PDF / HWP / XLSX / DOCX / ZIP)
       │
       ├──────────────────────────────┐
       ▼                              ▼
[Vector DB 구축]               [Graph DB 구축]
 ChromaDB                       Neo4j
 chunk_id 기반 인덱싱            Document → Chunk → Entity
 Upstage 임베딩                  chunk_id로 VectorDB와 연결
       │                              │
       └──────────────┬───────────────┘
                      ▼
             [Hybrid RAG 엔진]
              벡터 검색 → (선택) BM25+RRF → (선택) LLM 리랭크
              chunk_id 브리지 → Neo4j Entity 1-hop 확장
              확장 chunk 추가 조회 → 컨텍스트 병합 → solar-pro3 생성
                      │
                      ▼
                  [답변 출력]
```

### 그래프 구조

```
(Document) -[:HAS_CHUNK]->  (Chunk)  -[:MENTIONS]-> (Entity)
(Entity)   -[:REL_TYPE {evidence_chunk_id, source_doc_key}]-> (Entity)
(Entity)   -[:CO_OCCURS {count}]-> (Entity)
```

**chunk_id**는 ChromaDB와 Neo4j 양쪽에서 동일하게 사용되어 VectorDB ↔ GraphDB 브리지 역할을 합니다.

---

## 전처리 파이프라인

파이프라인은 번호 순서대로 실행합니다.

### Step 0 — 크롤러 (`pipeline/00_crawler.py`)

- `cse.knu.ac.kr` 공지사항 게시판을 BeautifulSoup으로 크롤링
- 공지 본문 + 첨부파일(PDF, HWP, XLSX 등)을 `data/raw/`, `data/attachments/`에 저장

### Step 1 — 파서 (`pipeline/01_parser.py`)

| 파일 형식 | 파싱 방법 |
|-----------|-----------|
| PDF | opendataloader-pdf v1.0+ `run()` API (JVM 기반 Markdown 변환) |
| HWP | LibreOffice 헤드리스 → 실패 시 pyhwp hwp5txt fallback |
| XLSX | pandas 멀티시트 텍스트 추출 |
| DOCX | python-docx 단락·표 추출 (병합 셀 중복 제거) |
| ZIP | 재귀 압축 해제 (zip-slip 방지 포함) |

파싱 결과는 `data/parsed/`에 JSON으로 저장되며, 이미 처리된 파일은 건너뜁니다(증분 처리).

**PDF 표 트랙 어노테이션 (`_annotate_track_sections`)**

opendataloader가 PDF 표를 선형화하면 여러 트랙의 요건이 한 텍스트에 혼재됩니다. 파싱 직후 `[다중전공트랙]`, `[해외복수학위트랙]`, `[학석사연계트랙]` 레이블을 각 `▸` 항목 앞에 삽입하여 LLM이 트랙을 정확히 구분할 수 있도록 합니다.

### Step 2 — Vector DB 구축 (`pipeline/02_vector_db.py`)

세 가지 변형 중 하나를 선택합니다.

| 스크립트 | 임베딩 모델 | Chroma 컬렉션 |
|----------|------------|--------------|
| `02_vector_db.py` | Upstage `embedding-passage` | `knu_cse_upstage_pro` |
| `02b_vector_db_bge.py` | `dragonkue/bge-m3-ko` (로컬) | `knu_cse_bge_ko` |
| `02c_vector_db_contextual.py` | Upstage + Anthropic Contextual Retrieval | `knu_cse_contextual` |

**chunk_id 형식** (03_graph_db.py와 동일 규칙 공유):

```
manual::{file_name}::chunk{i}
notice::{notice_num}::{att_name}::chunk{i}
notice_content::{notice_num}::chunk{i}
```

청킹: `RecursiveCharacterTextSplitter` (크기 800, 오버랩 150), 한국 행정 문서 구분자(`가/나/다`, `①②③` 등) 우선 적용.

> chunk_size 800, overlap 150은 03_graph_db.py와 동일하게 맞춰 **chunk_id 브리지 정합성**을 보장합니다. 표 구조 문서에서 트랙 요건이 같은 청크에 묶일 확률을 높이기 위해 기존 500/50에서 조정되었습니다.

### Step 3 — Graph DB 구축 (`pipeline/03_graph_db.py`)

**Document → Chunk → Entity** 3계층 구조로 Neo4j 지식 그래프를 구축합니다.

#### 구축 흐름

```
문서 로드 (manual + 공지 첨부파일 + 공지 본문)
    │
    ▼ 증분 처리 — graph_state.json 으로 변경된 파일만 처리
    │
    ▼ 문서를 800자 청크로 분할 (02_vector_db.py와 동일 — chunk_id 브리지 정합성)
    │
    ▼ [추출 레이어] 청크별 LLM 1회 호출 (solar-pro3)
       response_format=json_object, 엔티티 최대 15개, 관계 최대 20개
       경북대 도메인 스키마 주입 (8종 노드, 12종 관계)
       엔티티 name·type만 추출 (properties 제거 — 수치는 청크 본문에 보존)
       max_tokens 초과 시 완성된 객체만 부분 복구
       ALIAS_MAP 규칙 기반 정규화 (LLM 통합 없음)
    │
    ▼ [저장 레이어] Neo4j 저장
       Document, Chunk, Entity 노드
       HAS_CHUNK, MENTIONS, 의미 관계 엣지
       CO_OCCURS 자동 생성 (같은 청크에 함께 등장한 엔티티 쌍)
```

#### 실행 옵션

```bash
python pipeline/03_graph_db.py           # 증분 처리 (새 파일만)
python pipeline/03_graph_db.py --rebuild # 전체 재구축
```

#### 노드 설계

| 노드 | 주요 속성 |
|------|-----------|
| `Document` | `doc_key`, `file_name`, `source_type`, `date`, `notice_title`, `content_hash` |
| `Chunk` | `chunk_id` (ChromaDB와 1:1), `doc_key`, `chunk_index`, `text_preview` |
| `Entity` | `name`, `type` |

#### 도메인 노드 타입 (8종)

| 레이블 | 설명 |
|--------|------|
| `Department` | 학부·학과·단과대 |
| `Major` | 전공·이중전공 (복수전공, 부전공프로그램 등) |
| `Program` | 제도·프로그램 (조기졸업제도, 장학금 등) |
| `Course` | 강좌·교과목 |
| `Requirement` | 이수·졸업 요건 (학점, 평점 등) |
| `Period` | 기간·일정 |
| `Organization` | 외부 기관·기업 |
| `Person` | 교수·직원 |

#### 관계 타입 (12종)

`REQUIRES`, `HAS_CONDITION`, `HAS_EXCEPTION`,
`BELONGS_TO`, `OFFERS`, `APPLIES_TO`, `HAS_DEADLINE`,
`EXCLUDES`, `SUBSTITUTES_FOR`, `PART_OF`, `PROVIDES`, `RELATED_TO`

의미 관계에는 `evidence_chunk_id`, `source_doc_key`가 함께 저장되어 근거 추적이 가능합니다.
`CO_OCCURS` 관계는 저장 단계에서 자동 생성되며 `count` 속성으로 공출현 빈도를 누적합니다.

---

## Hybrid RAG 엔진

`pipeline/04_hybrid_rag.py`

### 검색 파이프라인

```
쿼리
  │
  ├─ Dense 검색 (Chroma, top-15)
  │   └─ BM25Okapi + RRF 융합 (기본 ON, USE_BM25_HYBRID=1)
  │   └─ solar-pro2 LLM 리랭크 → top-5 (USE_RERANK=1)
  │       도메인 특화 프롬프트: 트랙 태그 인식, 수치 정확도 우선
  │
  └─ Chunk-Anchored Graph Search  (GRAPH_USAGE=anchored, 기본값)
       벡터 결과의 chunk_id → Neo4j Chunk 조회
       Chunk -[:MENTIONS]-> Entity (직접 연결 엔티티)
       Entity 1-hop 관계 확장 (12종 의미 관계)
       연관 Entity를 MENTIONS하는 추가 Chunk 수집 (동일 문서 우선)
       ChromaDB에서 추가 청크 조회 → vector_docs 병합
       ▶ 병합 후 전체 재 Rerank (그래프 확장 청크 노이즈 제거)
       ▶ Graph Gating: 본문에 없는 triple 제거 (USE_GRAPH_GATING=1)
```

### 컨텍스트 병합 및 생성

- 그래프 관계 트리플 → **"핵심 관계/조건 단서"** 블록
- 벡터 청크(원본 + 그래프 확장) → **"문서 본문"** 블록
- solar-pro3가 한국 행정 문서 규칙(본문 우선, 예외 조건 누락 금지, **트랙 태그 기반 값 구분**)으로 최종 생성

### GRAPH_USAGE 모드

| 값 | 설명 |
|----|------|
| `anchored` | Chunk-Anchored Graph Search (기본, 최고 성능) |
| `context` | 그래프 결과를 컨텍스트로만 추가 |
| `expansion` | 쿼리 확장용 그래프 사용 |
| `off` | 그래프 비활성화 (Vector Only) |

---

## 평가 시스템

### 벤치마크 데이터셋 (`evaluation/qa_dataset.json`)

110개의 질문-정답 쌍, 3가지 유형:

| 유형 | 설명 |
|------|------|
| 단일문서조회형 | 하나의 공지에서 사실 추출 |
| 조건해석형 | 예외·조건이 복잡한 규정 해석 |
| 다문서결합형 | 여러 문서를 교차 추론 |

군 복학생, 창업 준비생 등 5가지 학생 페르소나 설정.

### 평가 스크립트

- `evaluation/evaluate.py` — 단일 모델 평가, LLM 판사(solar-pro3)로 0/1 채점. 실행 시 평가 범위를 대화형으로 입력받아 부분 평가 가능
- `evaluation/evaluate_all_models.py` — 6가지 임베딩×LLM 조합 일괄 평가, API 실패 시 지수 백오프 재시도, 체크포인트 기반 재개 지원

결과는 색상 코딩된 Excel(`.xlsx`)과 JSON으로 저장됩니다.

---

## 설치 및 환경 구성

### 사전 요구사항

- Python 3.10+
- Java 11+ (opendataloader-pdf 필요)
- LibreOffice (HWP 파싱 필요, PATH 등록 필요)
- Docker (Neo4j 실행)

### 패키지 설치

```bash
pip install -r requirements.txt
# BGE 임베딩 사용 시 추가
pip install sentence-transformers
# BM25 하이브리드 사용 시 추가
pip install rank-bm25
```

> `httpx<0.28` 핀은 `requirements.txt`에 포함되어 있습니다 (openai SDK 호환성).

### Neo4j 실행

```bash
docker compose up -d
```

Neo4j 브라우저: http://localhost:7474 (기본 계정: `neo4j` / `password1234`)

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
# LLM API 키 (사용하는 모델에 맞게 설정)
UPSTAGE_API_KEY=your_upstage_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password1234

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db

# 기본 생성 모델
DEFAULT_LLM=solar-pro3
```

---

## 실행 방법

### 1. 데이터 수집 및 인덱스 구축 (최초 1회)

```bash
# 크롤링
python pipeline/00_crawler.py

# 파싱
python pipeline/01_parser.py

# Vector DB 구축
python pipeline/02_vector_db.py

# Graph DB 구축 (증분)
python pipeline/03_graph_db.py

# Graph DB 전체 재구축
python pipeline/03_graph_db.py --rebuild
```

### 2. 질의응답

```bash
python pipeline/04_hybrid_rag.py
```

### 3. 평가 실행

```bash
# 단일 모델
python evaluation/evaluate.py

# 전체 모델 비교
python evaluation/evaluate_all_models.py
```

`evaluate.py` 실행 시 평가 범위를 대화형으로 선택합니다.

```
평가 범위 선택  (QA 데이터셋 총 110개 항목)
  입력 형식:
    0      전체 평가 (1~110번, 110개 모두)
    N      N번 항목 1개만        예) 5
    N,M    N번~M번 범위 평가     예) 1,50  →  1~50번
    N,     N번부터 끝까지        예) 51,   →  51~110번
    ,M     처음부터 M번까지      예) ,30   →  1~30번
  입력 >
```

---

## 환경 변수 레퍼런스

`04_hybrid_rag.py`의 동작은 환경 변수로 제어합니다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `EMBED_BACKEND` | `upstage` | 임베딩 백엔드: `upstage` / `bge` / `contextual` |
| `GRAPH_USAGE` | `anchored` | 그래프 활용 방식: `anchored` / `context` / `expansion` / `off` |
| `USE_RERANK` | `1` | LLM 리랭크 사용 여부 |
| `USE_BM25_HYBRID` | `1` | BM25 하이브리드 검색 사용 여부 (Anthropic: +RRF → 실패율 -49%) |
| `USE_GRAPH_GATING` | `1` | 그래프 노이즈 게이팅 (본문에 없는 triple 제거) |
| `USE_VERIFICATION` | `0` | Chain-of-Verification(CoV) 사용 여부 |
| `HYBRID_V3` | `0` | 실험적 LightRAG/PathRAG/CRAG 그래프 검색 사용 여부 |
| `RETRIEVE_K` | `15` | 벡터 검색 상위 K (reranker 후보 풀) |
| `BM25_K` | `15` | BM25 검색 상위 K |
| `RERANK_TO_K` | `5` | 리랭크 후 유지 상위 K |

---

## 디렉토리 구조

```
KNU/
├── pipeline/
│   ├── 00_crawler.py               # 게시판 크롤러
│   ├── 01_parser.py                # 다형식 문서 파서
│   ├── 02_vector_db.py             # Vector DB 구축 (Upstage)
│   ├── 02b_vector_db_bge.py        # Vector DB 구축 (BGE-m3-ko)
│   ├── 02c_vector_db_contextual.py # Vector DB 구축 (Contextual Retrieval)
│   ├── 03_graph_db.py              # Neo4j 지식 그래프 구축
│   ├── 03_graph_db_backup.py       # 이전 버전 백업
│   ├── 04_hybrid_rag.py            # Hybrid RAG 엔진 (메인)
│   ├── build_graph_embeddings.py   # 그래프 노드 임베딩 (레거시)
│   └── build_relation_embeddings.py # 그래프 관계 임베딩 (레거시)
│
├── evaluation/
│   ├── evaluate.py                 # 단일 모델 평가
│   ├── evaluate_all_models.py      # 전체 모델 일괄 평가
│   └── qa_dataset.json             # 110개 벤치마크 Q&A
│
├── data/
│   ├── raw/                        # 크롤링 원본 (notices.json)
│   ├── parsed/                     # 파싱 결과 JSON
│   ├── attachments/                # 다운로드된 첨부파일
│   ├── manual_files/               # 수동 추가 문서
│   └── graph_state.json            # 그래프 증분 처리 상태
│
├── docs/
│   ├── ablation-2hop.md            # Ablation 실험 기록
│   └── graph_db_redesign.md        # Graph DB 재설계 검토 문서
│
├── docker-compose.yml              # Neo4j 컨테이너 설정
├── requirements.txt
├── make_dataset.py                 # 벤치마크 데이터셋 생성 스크립트
└── .env                            # API 키 및 설정 (git 제외)
```

---

## 주요 설계 결정 및 Ablation 결과

코드 및 `docs/ablation-2hop.md`에 기록된 실험 결과입니다.

| 실험 | 결과 | 결정 |
|------|------|------|
| 청크 크기 250 vs 500 | 250이 -9점 열세 | 500 → 현재 **800** (표 구조 혼재 완화 + chunk_id 브리지 정합) |
| 2-hop 그래프 탐색 | -3점, 5.6배 느림 | 1-hop 유지 |
| BGE-m3-ko 임베딩 | -6점 (이 도메인에서) | Upstage 유지 |
| Contextual Retrieval | 67% 실패율 | 미채택 |
| 그래프 노이즈 게이팅(Phase 5) | 결론 불분명이었으나 재활성화 | **ON** (cross-doc 확장 노이즈 제거 후 유효) |
| GRAPH_USAGE=anchored vs context/expansion | anchored 최고 성능 | anchored 기본값 |
| 요약 후 추출 vs 직접 추출 | 요약 단계에서 조건·예외 관계 손실 | 요약 제거, 원문 직접 추출 |
| 문서 단위 추출 vs 청크 단위 추출 | 문서 단위는 6000자 트런케이션으로 정보 손실 | 1500자 청크 단위 추출 |

---

## 기술 스택 요약

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.10+ |
| 벡터 DB | ChromaDB 0.6 |
| 그래프 DB | Neo4j 5.20 (Docker) |
| LLM (생성) | Upstage solar-pro3 |
| LLM (리랭크/판사) | Upstage solar-pro3 |
| 임베딩 | Upstage embedding-passage (기본) |
| LLM 프레임워크 | LangChain 0.3 |
| 문서 파싱 | opendataloader-pdf, pyhwp, LibreOffice |
| 평가 | RAGAS, LLM-as-Judge, openpyxl |
| 인프라 | Docker Compose |

---

## 변경 이력

### `pipeline/01_parser.py`

| 항목 | 내용 |
|------|------|
| **PDF 표 트랙 어노테이션** | `_annotate_track_sections()` 추가 — opendataloader 선형화로 혼재된 트랙 요건에 `[트랙명]` 태그 삽입 |
| **opendataloader API 교체** | `convert()` → `run(generate_markdown=True)` (v1.0+ 호환) |
| **DOCX 병합 셀 중복 제거** | `python-docx` 병합 셀 중복 출력 방지 |
| **공지 단위 중간 저장** | 파싱 중 크래시 시 이전 공지 결과 보존 |

### `pipeline/02_vector_db.py`

| 항목 | 내용 |
|------|------|
| chunk_id 형식 변경 | `d{idx}::{file_name}::chunk{i}` → `{source_type}::{notice_num}::{file_name}::chunk{i}` (순서 독립적, Neo4j와 공유) |
| `make_doc_key()` / `make_chunk_id()` 추가 | 03_graph_db.py와 동일 규칙 공유 |
| 공지 본문(`content`) 인덱싱 추가 | 파싱 실패 첨부파일 25개의 정보 손실 방지 |
| 공지 제목 prepend | `[제목]\n본문` 형태로 임베딩 시 제목 의미 반영 |
| 짧은 청크 필터 | 20자 미만 청크 제거 (노이즈 방지) |
| 날짜 메타데이터 추가 | `date`, `notice_num` 필드 metadata 포함 |
| 중복 공지 제거 | `notice_num` 기준 중복 공지 스킵 |
| `list_collections()` 버그 수정 | `try/except` 방식으로 안전한 컬렉션 삭제 |
| **청크 크기 조정** | 500/50 → **800/150** (표 구조 트랙 혼재 완화, 03_graph_db.py와 정합) |

### `pipeline/03_graph_db.py`

| 항목 | 내용 |
|------|------|
| 그래프 구조 전면 재설계 | `Document → Entity` → **`Document → Chunk → Entity`** 3계층 구조 |
| chunk_id 브리지 | Chunk 노드의 `chunk_id`가 ChromaDB chunk_id와 1:1 매핑 |
| 증분 처리 | `data/graph_state.json`으로 변경된 파일만 처리, `--rebuild`로 전체 재구축 |
| 요약 단계 제거 | `summarize_document()` 삭제 → API 호출 절반 감소, 조건·예외 관계 손실 방지 |
| 청크 단위 추출 | 문서 단위 → **1500자 청크** 단위 추출 (더 많은 엔티티 추출) |
| 공지 본문 처리 추가 | `content` 필드를 별도 문서로 그래프 구축에 포함 |
| 관계 근거 저장 | 관계에 `evidence_chunk_id`, `source_doc_key` 저장 (근거 추적 가능) |
| 중복 공지 제거 | `notice_num` 기준 중복 스킵 |
| **LLM 통합 완전 제거** | `run_consolidation` 삭제 → ALIAS_MAP(30개) 규칙 기반 정규화로 교체. 문서당 추가 LLM 호출 0회 |
| **경북대 도메인 스키마** | 범용 17종 관계 → 도메인 특화 8종 노드 · 10종 관계로 축소 |
| **`response_format=json_object`** | LLM 출력 JSON 강제 → 마크다운 파싱 실패 제거 |
| **엔티티·관계 상한** | 프롬프트 명시 + 후처리 슬라이싱 (엔티티 ≤ 15, 관계 ≤ 20) |
| **CO_OCCURS 자동 생성** | 저장 단계에서 같은 청크 내 엔티티 쌍에 `CO_OCCURS {count}` 관계 생성 |
| **`ensure_constraints()`** | 빌드 시작 시 유니크 제약 생성 → Neo4j MERGE 성능 O(log n) |
| **`max_tokens` 4096으로 복구** | 2048로 줄이면 표 많은 청크에서 JSON 잘림(JSONDecodeError) 발생. 노이즈는 엔티티·관계 상한으로 제어하므로 token 제한 불필요 |
| **`write_entity` props 보호** | `SET e += $props` 전에 `name`·`type` 예약 키 제거 → 유니크 제약 키 덮어쓰기 방지 |
| **chunk_size를 vector DB와 통일** | 1500 → **800** (chunk_id 브리지 정합성 확보. 이전엔 vector=500, graph=1500 불일치로 브리지 파손) |
| **HAS_CONDITION·HAS_EXCEPTION 관계 복구** | 조건/예외 질문에 필요한 두 타입을 도메인 스키마에 추가 (10종 → 12종) |
| **엔티티 properties 제거** | 그래프는 탐색(네비게이터) 역할만 — 수치 데이터는 청크 본문에 보존. 출력 토큰 ~40% 감소, 잘림 오류 해소 |
| **`_recover_partial_json()`** | max_tokens 초과 시 완성된 JSON 객체만 부분 복구하여 데이터 손실 최소화 |

### `pipeline/04_hybrid_rag.py`

| 항목 | 내용 |
|------|------|
| `_raw_vector_search` 반환값 추가 | `chunk_id`, `doc_key` 필드 추가 (Graph 브리지용) |
| `chunk_anchored_graph_search()` 추가 | chunk_id → Neo4j Chunk → Entity 1-hop 확장 → 추가 chunk_id 반환 |
| `fetch_chunks_by_ids()` 추가 | 확장된 chunk_id로 ChromaDB 직접 조회 |
| anchored 모드 검색 흐름 개선 | 그래프 확장으로 발견된 추가 청크를 `vector_docs`에 자동 병합 |
| **BM25 기본 ON** | `USE_BM25_HYBRID` 기본값 0→1. Dense+BM25+RRF 조합 기본 적용 |
| **RETRIEVE_K 15로 확대** | reranker 후보 풀 10→15 (BM25 병행 시 후보 다양성 확보) |
| **그래프 확장 동일 문서 우선** | `chunk_anchored_graph_search` Step 4에 doc_key 정렬 추가 → cross-document 오염 방지 |
| **그래프 확장 후 통합 Rerank** | 그래프 확장 청크 추가 후 전체 vector_docs 재 Rerank → 무관 문서 필터링 |
| **Graph Gating 기본 ON** | `USE_GRAPH_GATING` 기본값 0→1. 본문에 없는 triple 제거 |
| **STRICT_GATING_RELS 확장** | `HAS_CONDITION` 추가 — 조건 관계도 양쪽 엔티티 본문 확인 |
| **Rerank 프롬프트 도메인 특화** | 트랙 태그 인식, 수치·조건 정확도 우선 순위 기준 추가 |
| **Rerank 콘텐츠 500→800자** | 더 많은 문맥으로 정확한 청크 순위 결정 |
| **생성 프롬프트 트랙 구분 규칙** | `[트랙명]` 태그 기반 값 구분 규칙 추가 (다른 트랙 값 혼용 금지) |
| **`GENERATION_TEMPERATURE` 0.2 → 0** | RAG 환경에서 temperature 0.2는 hallucination 유발 및 재현성 훼손. 0으로 변경하여 결정성 확보 |

### `evaluation/evaluate.py`

| 항목 | 내용 |
|------|------|
| **Judge 프롬프트 개선** | 추가 정보(정답보다 상세한 답변)로 인한 오판정 제거. 핵심 사실 누락·오류만 incorrect |
| **Ctrl+C 중단 시 저장** | `KeyboardInterrupt` 캐치 → 진행된 결과 저장 후 종료 |
| **메타데이터 수정** | `max_relations` 5→3, `n_results` 3→5 (실제 코드값과 일치) |
| **Judge 모델 solar-pro2 → solar-pro3** | LLM-as-Judge 원칙상 채점 모델은 생성 모델과 동급 이상이어야 판정 신뢰도 확보 가능. solar-pro2는 solar-pro3이 생성한 복잡한 답변을 과소 평가하는 경향 존재 |
| **부분 평가 범위 선택 기능** | 실행 시 `_prompt_eval_range()`로 범위를 대화형 입력받아 부분 평가 지원. `0`=전체, `N`=단일, `N,M`=범위, `N,`=N번~끝, `,M`=처음~M번 |
