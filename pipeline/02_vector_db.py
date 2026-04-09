import os
import json
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 경로 설정 ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# .env 명시적으로 로드
load_dotenv(ENV_PATH)

# ── 실험 조합 6개 ─────────────────────────────────────
EXPERIMENTS = [
    {
        "id": "openai_large",
        "provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "llm_model": "gpt-4o",
        "label": "OpenAI 대형 (text-embedding-3-large + gpt-4o)"
    },
    {
        "id": "openai_small",
        "provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "label": "OpenAI 소형 (text-embedding-3-small + gpt-4o-mini)"
    },
    {
        "id": "gemini_pro",
        "provider": "gemini",
        "embedding_model": "models/gemini-embedding-001",
        "llm_model": "gemini-1.5-pro",
        "label": "Gemini 대형 (gemini-embedding-001 + gemini-1.5-pro)"
    },
    {
        "id": "gemini_flash",
        "provider": "gemini",
        "embedding_model": "models/gemini-embedding-001",
        "llm_model": "gemini-1.5-flash",
        "label": "Gemini 소형 (gemini-embedding-001 + gemini-1.5-flash)"
    },
    {
        "id": "upstage_pro",
        "provider": "upstage",
        "embedding_model": "solar-embedding-1-large-passage",
        "llm_model": "solar-pro",
        "label": "Upstage 대형 (solar-embedding-passage + solar-pro)"
    },
    {
        "id": "upstage_mini",
        "provider": "upstage",
        "embedding_model": "solar-embedding-1-large-passage",
        "llm_model": "solar-mini",
        "label": "Upstage 소형 (solar-embedding-passage + solar-mini)"
    },
]


def get_embedding_function(experiment):
    provider = experiment["provider"]
    model = experiment["embedding_model"]

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
        print(f"  [임베딩] OpenAI: {model}")
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key
        )

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY가 .env에 없습니다.")
        print(f"  [임베딩] Gemini: {model}")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
            task_type="RETRIEVAL_DOCUMENT"
        )

    elif provider == "upstage":
        from langchain_upstage import UpstageEmbeddings
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY가 .env에 없습니다.")
        print(f"  [임베딩] Upstage: {model}")
        return UpstageEmbeddings(
            model=model,
            api_key=api_key
        )

    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")


def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def load_parsed_data():
    """파싱된 데이터 로드 - manual_files만 사용"""
    documents = []
    manual_path = os.path.join(PARSED_DIR, "manual_parsed.json")

    if os.path.exists(manual_path):
        with open(manual_path, encoding="utf-8") as f:
            manual_files = json.load(f)

        for mf in manual_files:
            parsed_text = mf.get("parsed_text", "").strip()
            if parsed_text:
                documents.append({
                    "text": f"[파일명] {mf['file_name']}\n[내용]\n{parsed_text}",
                    "metadata": {
                        "source": "manual",
                        "file_name": mf["file_name"]
                    }
                })

    print(f"[로드] 총 {len(documents)}개 문서 로드됨")
    return documents


def chunk_documents(documents):
    """문서를 청크 단위로 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i
                }
            })

    print(f"[청킹] {len(documents)}개 문서 → {len(chunks)}개 청크")
    return chunks


def get_existing_file_names(collection) -> set:
    """컬렉션에 이미 저장된 file_name 목록 반환"""
    try:
        result = collection.get(include=["metadatas"])
        return {m.get("file_name", "") for m in result.get("metadatas", []) if m.get("file_name")}
    except Exception:
        return set()


def build_vector_db(experiment, chunks, force_rebuild=False):
    """실험 조합 하나에 대해 Vector DB 구축 (중복 문서 자동 스킵)"""
    exp_id = experiment["id"]
    collection_name = f"knu_cse_{exp_id}"

    client = get_chroma_client()
    existing_collections = client.list_collections()  # chromadb 0.6.x에서는 이름 리스트 반환

    if collection_name in existing_collections:
        if force_rebuild:
            client.delete_collection(collection_name)
            print(f"  [초기화] 기존 컬렉션 삭제: {collection_name}")
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            already_embedded = set()
        else:
            # 기존 컬렉션 유지 — 이미 임베딩된 파일은 스킵
            collection = client.get_collection(collection_name)
            already_embedded = get_existing_file_names(collection)
            if already_embedded:
                print(f"  [중복 감지] 이미 임베딩된 파일 {len(already_embedded)}개 → 해당 청크 스킵")
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        already_embedded = set()

    # 이미 임베딩된 파일의 청크 제외
    new_chunks = [c for c in chunks if c["metadata"].get("file_name", "") not in already_embedded]

    if not new_chunks:
        print(f"  [스킵] 추가할 새 청크 없음 (전체 {len(chunks)}개 이미 존재)")
        return

    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  [스킵] {skipped}개 청크 제외 (중복 파일) → {len(new_chunks)}개 신규 청크 임베딩")

    embedding_fn = get_embedding_function(experiment)

    # 기존 ID와 충돌 방지를 위해 현재 컬렉션 크기 기준으로 오프셋 계산
    id_offset = collection.count()
    batch_size = 50
    total = len(new_chunks)

    for i in range(0, total, batch_size):
        batch = new_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"{exp_id}_chunk_{id_offset + i + j}" for j in range(len(batch))]

        embeddings = embedding_fn.embed_documents(texts)
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  [{min(i + batch_size, total)}/{total}] 저장 완료")

    print(f"  [완료] {collection_name} → 총 {collection.count()}개 청크")


def build_all(force_rebuild=False):
    """6개 실험 조합 Vector DB 전부 구축"""
    documents = load_parsed_data()
    if not documents:
        print("[오류] 파싱된 데이터가 없어요!")
        return

    chunks = chunk_documents(documents)

    for exp in EXPERIMENTS:
        print(f"\n[실험] {exp['label']}")
        try:
            build_vector_db(exp, chunks, force_rebuild=force_rebuild)
        except Exception as e:
            print(f"  [오류] {exp['id']}: {e}")

    print("\n[완료] 전체 Vector DB 구축 완료!")


def search(exp_id, query, n_results=3):
    """특정 실험 조합으로 검색"""
    exp = next((e for e in EXPERIMENTS if e["id"] == exp_id), None)
    if not exp:
        print(f"실험 ID를 찾을 수 없어요: {exp_id}")
        return []

    embedding_fn = get_embedding_function(exp)
    client = get_chroma_client()
    collection = client.get_collection(f"knu_cse_{exp_id}")

    query_embedding = embedding_fn.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "content": doc,
            "source": meta.get("file_name", ""),
            "score": round(1 - dist, 3)
        })

    return docs


if __name__ == "__main__":
    # 기존 컬렉션 삭제 후 재구축
    build_all(force_rebuild=True)