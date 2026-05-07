"""Graph 노드 임베딩 빌드 — semantic GraphRAG 검색용.

Neo4j 의 Entity 노드들을 텍스트로 표현해 Upstage embedding-passage 로
임베딩 → Chroma 별도 collection 'knu_cse_graph_nodes' 에 저장.

이전 graph_search 는 keyword CONTAINS 또는 vector_docs anchor 기반이라
한국어 동의어/약어 매칭에 약했음. 노드 임베딩으로 의미 검색 가능하게
변경하는 것이 정통 GraphRAG 패턴.

실행: python pipeline/build_graph_embeddings.py
"""
import os
import sys

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from langchain_upstage import UpstageEmbeddings

# Windows cp949 콘솔 대응
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
load_dotenv(ENV_PATH)

EMBEDDING_MODEL = "embedding-passage"  # 노드 인덱싱용 (vector DB 와 동일)
COLLECTION_NAME = "knu_cse_graph_nodes"
BATCH_SIZE = 64

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise RuntimeError("UPSTAGE_API_KEY 가 .env 에 없음")


def fetch_entity_nodes() -> list[dict]:
    """Neo4j 에서 Entity 노드 (Document 제외) 추출.
    각 노드: id, name, label, properties (source_files 제외), source_files."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    nodes = []
    try:
        with driver.session() as session:
            rows = session.run("""
                MATCH (n)
                WHERE NOT n:Document
                  AND coalesce(n.name, '') <> ''
                RETURN
                    elementId(n) AS id,
                    coalesce(n.name, '') AS name,
                    [l IN labels(n) WHERE l <> 'Entity'] AS labels,
                    properties(n) AS props
            """).data()
            for r in rows:
                props = r.get("props", {}) or {}
                # source_files 는 검색용으로 별도 분리, name 은 이미 갖고 있음
                source_files = props.pop("source_files", [])
                # name 도 props 에서 제거 (중복 방지)
                props.pop("name", None)
                nodes.append({
                    "id": r["id"],
                    "name": r["name"].strip(),
                    "labels": r.get("labels", []),
                    "properties": props,
                    "source_files": source_files,
                })
    finally:
        driver.close()
    return nodes


def node_to_text(node: dict) -> str:
    """노드를 임베딩용 텍스트로 변환.
    형식: '{name} (type: {label}) | property: k=v, k=v | 출처: file1, file2'"""
    parts = [node["name"]]

    # type label (있을 때)
    if node["labels"]:
        labels_str = ", ".join(node["labels"][:3])  # 다중 label 시 최대 3개
        parts.append(f"(type: {labels_str})")

    # properties — name/source_files 제외 (이미 분리)
    if node["properties"]:
        prop_strs = []
        for k, v in list(node["properties"].items())[:8]:  # 최대 8개
            if v is None or str(v).strip() == "":
                continue
            v_str = str(v)[:120]  # 너무 긴 값 자름
            prop_strs.append(f"{k}={v_str}")
        if prop_strs:
            parts.append("| properties: " + ", ".join(prop_strs))

    # 출처 파일 (간략하게 처음 2개만, 의미 보강용)
    if node["source_files"]:
        files_str = ", ".join(f[:50] for f in node["source_files"][:2])
        parts.append(f"| 출처: {files_str}")

    return " ".join(parts)


def main():
    print(f"[설정] embedding={EMBEDDING_MODEL}, collection={COLLECTION_NAME}")

    nodes = fetch_entity_nodes()
    print(f"\n[Neo4j] Entity 노드 {len(nodes)}개 추출")

    if not nodes:
        print("[경고] 노드 없음 — graph DB 빌드 먼저 필요")
        return

    # 텍스트 변환
    texts = [node_to_text(n) for n in nodes]
    sample = texts[0] if texts else ""
    print(f"  샘플: {sample[:150]}")

    # 임베딩
    embedding_fn = UpstageEmbeddings(model=EMBEDDING_MODEL, api_key=UPSTAGE_API_KEY)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )

    # 기존 collection 삭제 후 재생성
    if COLLECTION_NAME in list(client.list_collections()):
        print(f"[초기화] 기존 컬렉션 '{COLLECTION_NAME}' 삭제")
        client.delete_collection(COLLECTION_NAME)
    coll = client.create_collection(COLLECTION_NAME)

    print(f"[임베딩] {len(nodes)}개 노드 batch={BATCH_SIZE}")
    for start in range(0, len(nodes), BATCH_SIZE):
        batch_nodes = nodes[start:start + BATCH_SIZE]
        batch_texts = texts[start:start + BATCH_SIZE]
        embeddings = embedding_fn.embed_documents(batch_texts)
        coll.add(
            # node id 는 elementId (Neo4j 내부 id) 라 unique 보장
            ids=[n["id"] for n in batch_nodes],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[
                {
                    "name": n["name"],
                    "label": (n["labels"] or ["Entity"])[0],
                }
                for n in batch_nodes
            ],
        )
        print(f"  batch {start // BATCH_SIZE + 1}: {start+1}-{start+len(batch_nodes)} / {len(nodes)}")

    print(f"\n[완료] '{COLLECTION_NAME}' 컬렉션 {coll.count()}개 노드 임베딩")


if __name__ == "__main__":
    main()
