"""Vector DB 빌드 — Anthropic Contextual Retrieval (2024-09).

각 chunk 앞에 LLM 이 생성한 doc-level 컨텍스트 (50-100 token) 를 prepend
한 후 임베딩. chunk 가 단독으로 의미 모호한 경우 (예: "신청 기한은 6월
15일") 에 "이 chunk 는 X 공지의 신청 안내 부분" 같은 컨텍스트가 매칭
정확도 ↑.

근거 — 출처: https://www.anthropic.com/news/contextual-retrieval
- 실패율 -35% (embeddings only) / -49% (with BM25) / -67% (with rerank)
- 우리는 이미 BM25 (USE_BM25_HYBRID) + LLM rerank (USE_RERANK) 보유
  → 누적 적용 시 -67% 영역 기대

Cost: 602 chunk × 1 LLM call ≈ ~$0.5 (solar-pro2 short prompts).

별도 collection 'knu_cse_upstage_contextual' 사용 → 기존 'knu_cse_upstage_pro'
유지 (rollback 용 + ablation 비교).

실행:
  python pipeline/02c_vector_db_contextual.py
  EMBED_BACKEND=contextual python evaluation/evaluate.py
"""
import os
import sys
import json

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_upstage import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
load_dotenv(ENV_PATH)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "embedding-passage"
COLLECTION_NAME = "knu_cse_upstage_contextual"
CONTEXT_MODEL = "solar-pro2"  # context 생성용 (cheap, fast)
BATCH_SIZE = 64
DOC_TRUNCATE = 6000  # context 생성 시 doc 본문 truncate (LLM context 보호)

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise RuntimeError("UPSTAGE_API_KEY 가 .env 에 없음")

upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")


def load_documents() -> list[dict]:
    docs = []
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")
    if os.path.exists(manual_path):
        with open(manual_path, encoding="utf-8") as f:
            for m in json.load(f):
                text = (m.get("parsed_text") or "").strip()
                if text:
                    docs.append({
                        "file_name": m.get("file_name", "unknown"),
                        "source_type": "manual",
                        "parsed_text": text,
                    })
    notices_path = os.path.join(PARSED_DIR, "notices_parsed.json")
    if os.path.exists(notices_path):
        with open(notices_path, encoding="utf-8") as f:
            for n in json.load(f):
                title = (n.get("title") or "").strip()
                for a in n.get("attachments", []):
                    text = (a.get("parsed_text") or "").strip()
                    if not text:
                        continue
                    docs.append({
                        "file_name": a.get("name", "unknown"),
                        "source_type": "notice",
                        "notice_title": title,
                        "parsed_text": text,
                    })
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """02_vector_db.py 와 동일 한국어 separator. 단 chunk 마다 doc_index +
    full_doc_text 포함 → context 생성 단계에서 doc 본문 참조 가능."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n## ", "\n# ",
            "\n가. ", "\n나. ", "\n다. ", "\n라. ", "\n마. ",
            "\n1) ", "\n2) ", "\n3) ", "\n4) ", "\n5) ",
            "\n① ", "\n② ", "\n③ ", "\n④ ", "\n⑤ ",
            "\n- ", "\n* ",
            "\n",
            ". ", "。", "! ", "? ",
            " ", ""
        ],
        length_function=len,
    )
    chunks = []
    for doc_idx, doc in enumerate(docs):
        text = doc["parsed_text"]
        splits = splitter.split_text(text)
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "id": f"d{doc_idx:04d}::{doc['file_name']}::chunk{i}",
                "text": chunk_text,
                "doc_text": text,  # context 생성용 (저장 X)
                "doc_idx": doc_idx,
                "metadata": {
                    "file_name": doc["file_name"],
                    "source_type": doc["source_type"],
                    "chunk_index": i,
                    "doc_index": doc_idx,
                    **({"notice_title": doc["notice_title"]}
                       if "notice_title" in doc else {}),
                },
            })
    return chunks


# Anthropic 원문 prompt 한국어 버전 — chunk 가 doc 의 어느 위치/주제인지
# 50~100 token 으로 짧게 설명. 검색 시 query 와 매칭 정확도 ↑.
CONTEXT_PROMPT_TEMPLATE = """다음은 전체 문서와 그 일부 chunk 입니다.

<전체 문서>
{document}
</전체 문서>

<chunk>
{chunk}
</chunk>

이 chunk 가 검색 (retrieval) 에 잘 매칭되도록, **전체 문서 안에서의
위치/주제를 짧게 설명**하세요. 50-100 토큰. 추가 설명 없이 컨텍스트
문장만 출력. chunk 내용 자체를 반복하지 말고 *그 chunk 가 무엇에 대한
부분인지* 만 작성.

예) "이 chunk 는 [공지명] 의 [신청 자격] 부분으로, [핵심 조건] 을 다룬다."

컨텍스트:"""


# 같은 doc 의 여러 chunk 처리 시 doc 본문은 캐싱 — 비용 절감.
_doc_text_cache = {}


def generate_context_for_chunk(chunk: dict) -> str:
    """LLM 으로 chunk 의 doc-level 컨텍스트 생성. 실패 시 빈 문자열 반환
    (해당 chunk 는 컨텍스트 prefix 없이 임베딩됨, 안전한 fallback)."""
    doc_text = chunk["doc_text"]
    if len(doc_text) > DOC_TRUNCATE:
        doc_text = doc_text[:DOC_TRUNCATE] + "...[중략]"

    prompt = CONTEXT_PROMPT_TEMPLATE.format(
        document=doc_text,
        chunk=chunk["text"],
    )
    try:
        r = upstage_client.chat.completions.create(
            model=CONTEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [context 생성 오류] chunk {chunk['id']}: {e}")
        return ""


def main():
    print(f"[설정] chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"[설정] embedding={EMBEDDING_MODEL}, collection={COLLECTION_NAME}")
    print(f"[설정] context_model={CONTEXT_MODEL}")

    docs = load_documents()
    print(f"\n[로드] 문서 {len(docs)}개")

    chunks = chunk_documents(docs)
    print(f"[청킹] 청크 {len(chunks)}개")
    if chunks:
        lens = [len(c["text"]) for c in chunks]
        print(f"  길이 평균 {sum(lens)//len(lens)}자, 최소 {min(lens)}, 최대 {max(lens)}")

    # ── 1단계: 각 chunk 에 LLM context 생성 ──
    print(f"\n[Context 생성] {len(chunks)} chunk × LLM 호출 (~ {len(chunks)*0.6/60:.1f} 분, ~$0.5)")
    contextualized = []
    for i, chunk in enumerate(chunks):
        ctx = generate_context_for_chunk(chunk)
        # contextualized text = "[컨텍스트] {ctx}\n\n{chunk}"
        if ctx:
            full_text = f"[컨텍스트] {ctx}\n\n{chunk['text']}"
        else:
            full_text = chunk["text"]
        contextualized.append({
            "id": chunk["id"],
            "text": full_text,
            "metadata": chunk["metadata"],
        })
        if (i + 1) % 50 == 0:
            print(f"  진행: {i+1}/{len(chunks)}")

    # ── 2단계: 임베딩 + Chroma 저장 ──
    embedding_fn = UpstageEmbeddings(model=EMBEDDING_MODEL, api_key=UPSTAGE_API_KEY)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    if COLLECTION_NAME in list(client.list_collections()):
        print(f"[초기화] 기존 컬렉션 '{COLLECTION_NAME}' 삭제")
        client.delete_collection(COLLECTION_NAME)
    coll = client.create_collection(COLLECTION_NAME)

    print(f"\n[임베딩] {len(contextualized)} contextualized chunk batch={BATCH_SIZE}")
    for start in range(0, len(contextualized), BATCH_SIZE):
        batch = contextualized[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedding_fn.embed_documents(texts)
        coll.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  batch {start // BATCH_SIZE + 1}: {start+1}-{start+len(batch)} / {len(contextualized)}")

    print(f"\n[완료] '{COLLECTION_NAME}' 컬렉션 {coll.count()}개 contextualized chunk")


if __name__ == "__main__":
    main()
