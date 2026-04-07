import os
import json
from dotenv import load_dotenv
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_PROVIDER = "openai"


def get_embedding_function(provider=EMBEDDING_PROVIDER):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        print("[임베딩] OpenAI text-embedding-3-small 사용")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("[임베딩] Gemini embedding-004 사용")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "upstage":
        from langchain_upstage import UpstageEmbeddings
        print("[임베딩] Upstage Solar 사용")
        return UpstageEmbeddings(
            model="solar-embedding-1-large",
            api_key=os.getenv("UPSTAGE_API_KEY")
        )
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}")


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
    """시맨틱 청킹"""
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
                "metadata": {**doc["metadata"], "chunk_index": i}
            })

    print(f"[청킹] {len(documents)}개 문서 → {len(chunks)}개 청크")
    return chunks


def build_vector_db(provider=EMBEDDING_PROVIDER):
    """Vector DB 구축"""
    print(f"\n[시작] Vector DB 구축 (임베딩: {provider})")

    documents = load_parsed_data()
    if not documents:
        print("[오류] 파싱된 데이터가 없어요!")
        return

    chunks = chunk_documents(documents)
    embedding_fn = get_embedding_function(provider)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = f"knu_cse_{provider}"

    # Chroma 0.6.x 호환 - list_collections()가 이름 목록 반환
    existing = client.list_collections()
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"[초기화] 기존 컬렉션 '{collection_name}' 삭제")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]

        embeddings = embedding_fn.embed_documents(texts)

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"  [{min(i+batch_size, total)}/{total}] 저장 완료")

    print(f"\n[완료] Vector DB 구축 완료!")
    print(f"컬렉션: {collection_name}")
    print(f"총 청크 수: {collection.count()}")


def search_test(query, provider=EMBEDDING_PROVIDER, n_results=3):
    """검색 테스트"""
    print(f"\n[검색 테스트] '{query}'")

    embedding_fn = get_embedding_function(provider)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = f"knu_cse_{provider}"
    collection = client.get_collection(collection_name)

    query_embedding = embedding_fn.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n--- 결과 {i+1} (유사도: {1-dist:.3f}) ---")
        print(f"출처: {meta.get('title') or meta.get('file_name', '')}")
        print(f"내용: {doc[:200]}")


if __name__ == "__main__":
    PROVIDER = "openai"

    build_vector_db(PROVIDER)

    search_test("졸업 요건", PROVIDER)
    search_test("장학금 신청 기간", PROVIDER)