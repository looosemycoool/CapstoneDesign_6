# CapstoneDesign_6

# 🔍 Hybrid RAG System

> 2026 SW중심대학 학부생 중심 산학협력 프로젝트  
> 협력 기업: **엔코아** | 멘토: 왕태웅 책임매니저

---

## 📌 프로젝트 개요

Vector DB 기반의 기존 RAG 시스템은 단순 유사도 검색에는 강하지만, 복잡한 맥락 추론과 인과관계 파악에 한계가 있습니다.  
본 프로젝트는 **Vector DB + Graph DB를 결합한 Hybrid RAG 시스템**을 구축하여, 보다 정교하고 논리적인 문서 검색 및 답변 생성을 목표로 합니다.

---

## 🎯 목표

- 다양한 형태의 문서를 AI가 이해할 수 있는 형태로 변환하는 전처리 파이프라인 구축
- Vector DB + Graph DB 기반의 Hybrid RAG 시스템 구현
- Agentic RAG를 통한 질문 의미 추론 및 정교한 검색 결과 제공

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| Language | Python 3.10+ |
| RAG Framework | LangChain / LlamaIndex |
| Vector DB | Chroma / Weaviate |
| Graph DB | Neo4j |
| LLM | OpenAI GPT-4o / Claude API |
| Embedding | text-embedding-3-small / bge-m3 |
| Backend | FastAPI |
| Frontend | Streamlit / React |
| 평가 | RAGAS |
| 인프라 | Docker |

---

## 🗂️ 프로젝트 구조

```
hybrid-rag/
├── data/                  # 원본 문서 및 전처리 데이터
├── preprocessing/         # 문서 전처리 모듈 (청크 분할, 임베딩)
├── vector_db/             # Vector DB 구축 및 검색 모듈
├── graph_db/              # Graph DB 스키마 및 쿼리 모듈
├── rag_pipeline/          # Hybrid RAG 파이프라인 및 Agentic 로직
├── api/                   # FastAPI 백엔드
├── frontend/              # 사용자 인터페이스
├── evaluation/            # RAGAS 기반 성능 평가
├── docs/                  # 문서화
└── README.md
```

---

## 🏗️ 시스템 아키텍처

```
사용자 질문
     ↓
[Agentic Layer] ← 질문 분석 & 검색 전략 결정
     ↓                    ↓
[Vector DB 검색]    [Graph DB 검색]
     ↓                    ↓
      검색 결과 통합 (Re-ranking)
               ↓
        LLM에 Context 전달
               ↓
           최종 답변 생성
```

---

## 📅 진행 로드맵

| 단계 | 기간 | 내용 |
|------|------|------|
| 1단계 | 1~2주 | 환경 세팅, 기술 스택 확정, 데이터 도메인 결정 |
| 2단계 | 3~5주 | 문서 전처리 파이프라인, Vector RAG 베이스라인 구현 |
| 3단계 | 6~8주 | Graph DB 구축, Hybrid RAG 통합 |
| 4단계 | 9~11주 | Agentic RAG 적용, 성능 평가 및 개선 |
| 5단계 | 12주~ | 프론트엔드 연결, 최종 데모 및 발표 준비 |

---

## 👥 팀 구성 및 역할

| 역할 | 담당 내용 |
|------|-----------|
| 데이터/전처리 + Vector DB | 문서 수집, 청크 분할, 임베딩, Vector DB 구축 |
| Graph DB | Neo4j 스키마 설계, 개체-관계 추출, 쿼리 구현 |
| RAG 파이프라인 + Agentic | LangChain 기반 파이프라인, Agent 라우팅 로직 |
| Backend + Frontend | FastAPI, Streamlit/React UI |

---

## ⚙️ 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/{팀명}/hybrid-rag.git
cd hybrid-rag

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력

# Docker로 DB 실행
docker-compose up -d

# 실행
python main.py
```

---

## 📊 평가 지표

- **Faithfulness** — 생성된 답변이 검색된 문서에 근거하는 정도
- **Answer Relevancy** — 답변이 질문과 얼마나 관련 있는지
- **Context Recall** — 관련 문서를 얼마나 잘 검색했는지
- Vector RAG 단독 vs Hybrid RAG 비교 실험을 통한 성능 향상 수치 제시

---

## 🤝 협력 기관

**엔코아(En-Core)**  
멘토: 왕태웅 책임매니저 (twwang@gmail.com)
