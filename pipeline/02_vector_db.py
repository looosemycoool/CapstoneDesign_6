"""Vector DB 빌드 (Chroma) — manual + notices attachments 를 청킹해서 임베딩.

이전엔 이 파일이 graph DB 추출 코드로 repurpose 돼 있었으나 03_graph_db.py
와 중복 + 옛 버전이라 제거. 본래 의도(00 crawler → 01 parser → 02 vector
db → 03 graph db → 04 hybrid rag) 대로 vector DB 빌드 스크립트로 복원.

실행: python pipeline/02_vector_db.py
"""
import os
import sys
import json

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_upstage import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Windows cp949 콘솔 대응
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

# ── 청킹 / 임베딩 설정 (paper 기록용 단일 source of truth) ──
# chunk_size 250 ablation 결과 hybrid 정답률 -9 (52→43) 로 음의 효과 →
# 베스트 결과(20260505_201739) 와 동등한 chunk_size=500 으로 복원.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "embedding-passage"   # Upstage 현행 표기
COLLECTION_NAME = "knu_cse_upstage_pro"  # 평가가 참조하는 컬렉션
BATCH_SIZE = 64                          # 임베딩 API 호출 배치

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise RuntimeError("UPSTAGE_API_KEY 가 .env 에 없음")


def load_documents() -> list[dict]:
    """매뉴얼 + 공지 첨부파일을 단일 list 로 평탄화.
    각 항목: {file_name, source_type, parsed_text}.
    """
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
    """RecursiveCharacterTextSplitter 로 청크 분할. metadata 보존.
    한국어 문서를 위해 separator 우선순위: 줄바꿈 → 마침표/쉼표 → 공백.
    ID 는 doc_index 를 prefix 로 붙여 file_name 중복 시에도 unique 보장.
    (서로 다른 공지에 같은 첨부 파일명이 있는 경우가 있음.)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc_idx, doc in enumerate(docs):
        text = doc["parsed_text"]
        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": f"d{doc_idx:04d}::{doc['file_name']}::chunk{i}",
                "text": chunk_text,
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


def main():
    print(f"[설정] chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"[설정] embedding={EMBEDDING_MODEL}, collection={COLLECTION_NAME}")

    docs = load_documents()
    print(f"\n[로드] 문서 {len(docs)}개")

    chunks = chunk_documents(docs)
    print(f"[청킹] 청크 {len(chunks)}개")
    if chunks:
        lens = [len(c["text"]) for c in chunks]
        print(f"  길이 평균 {sum(lens)//len(lens)}자, 최소 {min(lens)}, 최대 {max(lens)}")

    embedding_fn = UpstageEmbeddings(model=EMBEDDING_MODEL, api_key=UPSTAGE_API_KEY)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    # 기존 컬렉션 삭제 후 재생성 (정확한 chunk_size 일관성 보장)
    if COLLECTION_NAME in list(client.list_collections()):
        print(f"[초기화] 기존 컬렉션 '{COLLECTION_NAME}' 삭제")
        client.delete_collection(COLLECTION_NAME)
    coll = client.create_collection(COLLECTION_NAME)

    print(f"[임베딩] {len(chunks)}개 chunk 를 batch={BATCH_SIZE} 로 처리")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedding_fn.embed_documents(texts)
        coll.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  batch {start // BATCH_SIZE + 1}: {start+1}-{start+len(batch)} / {len(chunks)}")

    print(f"\n[완료] '{COLLECTION_NAME}' 컬렉션 {coll.count()}개 chunk")


if __name__ == "__main__":
    main()
