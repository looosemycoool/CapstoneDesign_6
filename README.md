# Hybrid RAG Chatbot — 경북대 컴퓨터학부 공지 Q&A 시스템

> 2026 SW중심대학 학부생 중심 산학협력 프로젝트  

Vector DB(Chroma)와 Graph DB(Neo4j)를 결합한 Hybrid RAG 방식으로, 경북대학교 컴퓨터학부 공지사항에 대한 정확한 Q&A 답변을 생성합니다.

---

## 아키텍처

```
공지 크롤링 (00)
      ↓
문서 파싱 (01)  ←  PDF / HWP / TXT / XLSX
      ↓
 ┌────┴─────┐
 │          │
Vector DB  Graph DB       ← Chroma + Neo4j
(Chroma)  (Neo4j)
 │          │
 └────┬─────┘
      ↓
Hybrid RAG 답변 생성 (04)   ← OpenAI / Gemini / Upstage
      ↓
FastAPI 백엔드 + 프론트엔드
```

---

## 지원 모델 조합 (6가지)

| ID | 임베딩 모델 | LLM |
|----|------------|-----|
| `openai_large` | text-embedding-3-large | gpt-4o |
| `openai_small` | text-embedding-3-small | gpt-4o-mini |
| `gemini_pro` | models/embedding-001 | gemini-1.5-pro |
| `gemini_flash` | models/embedding-001 | gemini-1.5-flash |
| `upstage_pro` | solar-embedding-1-large | solar-pro |
| `upstage_mini` | solar-embedding-1-large | solar-mini |

---

## 사전 요구사항

- **Python 3.11** 이상
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Neo4j 컨테이너 실행용)
- API 키 (사용하는 모델에 맞춰 발급)
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Google AI Studio](https://aistudio.google.com/app/apikey) (Gemini)
  - [Upstage Console](https://console.upstage.ai/) (Solar)

---

## 환경 세팅

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/CapstoneDesign_6.git
cd CapstoneDesign_6
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 환경변수 설정 (.env)

프로젝트 루트에 `.env` 파일을 생성합니다.  
사용하지 않는 API는 비워둬도 됩니다.

```env
# API 키 (사용하는 모델에 해당하는 것만 입력)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
UPSTAGE_API_KEY=up_...

# Neo4j 연결 (docker-compose 기본값 그대로 사용 가능)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password1234

# Chroma 저장 경로
CHROMA_PERSIST_DIR=./chroma_db

# 기본 LLM
DEFAULT_LLM=gpt-4o-mini
```

> `.env`는 `.gitignore`에 포함되어 있어 git에 올라가지 않습니다.

### 4. Neo4j 실행 (Docker)

```bash
docker-compose up -d
```

실행 후 [http://localhost:7474](http://localhost:7474) 에서 Neo4j 브라우저에 접속할 수 있습니다.  
로그인: ID `neo4j` / PW `password1234`

---

## 파이프라인 실행

순서대로 실행합니다.

### Step 0 — 공지 크롤링

```bash
python pipeline/00_crawler.py
```

크롤링 결과는 `data/raw/notices.json`에 저장됩니다.

---

### Step 1 — 문서 파싱

```bash
python pipeline/01_parser.py
```

PDF, HWP, TXT, XLSX 첨부파일을 파싱해 `data/parsed/`에 JSON으로 저장합니다.

---

### Step 2 — Vector DB 구축

```bash
python pipeline/02_vector_db.py
```

- 6개 실험 조합에 대해 Chroma 컬렉션을 생성하고 청크를 임베딩합니다.
- **이미 임베딩된 파일은 자동으로 스킵**됩니다. 새 파일만 추가 임베딩하므로, 문서가 늘어날 때 전체를 다시 돌릴 필요가 없습니다.
- 결과는 `chroma_db/` 폴더에 영구 저장됩니다.

> 처음부터 다시 구축하려면 코드 마지막 줄을 `build_all(force_rebuild=True)`로 변경 후 실행하세요.

---

### Step 3 — Graph DB 구축

```bash
python pipeline/03_graph_db.py
```

- GPT-4o-mini가 각 문서에서 개체와 관계를 **동적으로** 추출합니다 (고정 스키마 없음).
- 전체 추출 완료 후 의미가 유사한 노드를 자동으로 **통합**합니다 (예: "졸업요건" / "졸업 요건" → 하나로).
- 완료 후 Neo4j 브라우저에서 확인:

```cypher
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50
```

---

### Step 4 — Hybrid RAG 테스트 (단일 쿼리)

```bash
python pipeline/04_hybrid_rag.py
```

코드 하단 `if __name__ == "__main__":` 블록의 쿼리를 수정해서 테스트해보세요.

---

## 평가 실행

### 단일 모델 평가 (gpt-4o-mini)

```bash
python evaluation/evaluate.py
```

### 전체 6개 모델 자동 평가

```bash
python evaluation/evaluate_all_models.py
```

- 결과는 `evaluation/results/` 폴더에 `.xlsx`와 `.json`으로 저장됩니다.
- 중간에 프로세스가 종료되어도 `evaluation/results/checkpoint/`에 체크포인트가 남아 재실행 시 이어서 진행합니다.

**xlsx 구성:**

| 시트 | 내용 |
|------|------|
| ① 요약 | 6개 모델 성능 비교 (성공률, 평균 점수) |
| ② Vector 비교 | 전체 질문에 대한 Vector Only 결과 6모델 나란히 |
| ③ Hybrid 비교 | 전체 질문에 대한 Hybrid RAG 결과 6모델 나란히 |
| ④~⑨ | 모델별 상세 결과 |

---

## 프로젝트 구조

```
CapstoneDesign_6/
├── pipeline/
│   ├── 00_crawler.py          # 공지 크롤링
│   ├── 01_parser.py           # 첨부파일 파싱 (PDF / HWP / TXT / XLSX)
│   ├── 02_vector_db.py        # Chroma Vector DB 구축 (중복 문서 자동 스킵)
│   ├── 03_graph_db.py         # Neo4j Graph DB 구축 (동적 추출 + 노드 통합)
│   └── 04_hybrid_rag.py       # Hybrid RAG 검색 및 답변 생성
│
├── evaluation/
│   ├── evaluate.py            # 단일 모델 평가 (gpt-4o-mini)
│   ├── evaluate_all_models.py # 6개 모델 전체 자동 평가
│   ├── qa_dataset.json        # 평가용 질문-정답 데이터셋
│   └── results/               # 평가 결과 (xlsx / json / checkpoint)
│
├── backend/                   # FastAPI 백엔드
├── frontend/                  # 프론트엔드
├── data/
│   ├── raw/                   # 크롤링 원본 JSON
│   ├── parsed/                # 파싱된 문서 JSON
│   ├── manual_files/          # 수동 추가 문서
│   └── attachments/           # 크롤링 첨부파일 원본
│
├── chroma_db/                 # Chroma 영구 저장소 (git 제외)
├── docker-compose.yml         # Neo4j Docker 설정
├── requirements.txt
└── .env                       # API 키 등 환경변수 (git 제외)
```

---

## 트러블슈팅

**Neo4j 연결 실패**  
Docker가 실행 중인지 확인하세요.
```bash
docker ps
# rag-neo4j 컨테이너가 없다면:
docker-compose up -d
```

**Chroma `file in use` 오류 (Windows)**  
여러 Python 프로세스가 같은 `chroma_db`를 동시에 열면 충돌합니다.  
다른 터미널에서 실행 중인 프로세스를 종료하고 재실행하세요.

**Rate Limit 오류 (임베딩 / LLM API)**  
`evaluate_all_models.py`는 지수 백오프 재시도 로직이 내장되어 있어 자동으로 대기 후 재시도합니다.  
그래도 반복된다면 `.env`의 API 키 플랜 한도를 확인하세요.

**xlsx 파일이 열리지 않는 경우**  
`evaluate.py`를 최신 버전으로 재실행하면, `_save_clean_xlsx()`가 boolean 값을 자동으로 문자열로 변환해 저장합니다.

---

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `chromadb 0.6.x` | Vector DB |
| `neo4j` | Graph DB 드라이버 |
| `langchain`, `langchain-openai`, `langchain-google-genai`, `langchain-upstage` | 임베딩 / LLM 연동 |
| `openai` | GPT API |
| `fastapi`, `uvicorn` | 백엔드 API 서버 |
| `pdfplumber` | PDF 파싱 |
| `openpyxl`, `pandas` | 평가 결과 Excel 출력 |
| `python-dotenv` | 환경변수 관리 |

전체 목록은 `requirements.txt` 참고.

---

## 라이선스

본 프로젝트는 경북대학교 컴퓨터학부 2026 캡스톤디자인 프로젝트입니다.
