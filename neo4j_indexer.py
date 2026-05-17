"""
Neo4j 인덱서
기본 스키마:
  (Document)-[:CONTAINS]->(Chunk)
  (Chunk)-[:MENTIONS]->(Entity)
  (Entity)-[:RELATED_TO {type}]->(Entity)

스키마 기반 확장 (ontology_discoverer 사용 시):
  (Chunk)-[:DEFINES|EXEMPLIFIES|DESCRIBES]->(DomainNode)
  (DomainNode)-[DOMAIN_REL]->(DomainNode)
"""
import logging
import math
import re
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver

from config import settings
from core.document_processor.chunker import Chunk
from core.entity_extractor import ExtractionResult

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
    return _driver


async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def ensure_constraints():
    """유니크 제약 및 인덱스 생성"""
    driver = get_driver()
    async with driver.session() as session:
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            # name만으로 유일 식별 — 복합키 사용 시 MATCH {name}만으로 조회 불가
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        for cypher in constraints:
            try:
                await session.run(cypher)
            except Exception as e:
                logger.debug(f"Constraint already exists or failed: {e}")


async def index_document(
    doc_id: str,
    filename: str,
    chunks: list[Chunk],
    extraction_results: list[ExtractionResult],
):
    """문서/청크/엔티티를 Neo4j에 저장"""
    driver = get_driver()
    async with driver.session() as session:
        # 문서 노드 생성
        await session.run(
            "MERGE (d:Document {id: $id}) SET d.filename = $filename, d.chunk_count = $count",
            id=doc_id, filename=filename, count=len(chunks),
        )

        for chunk, extraction in zip(chunks, extraction_results):
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"

            # 청크 노드 생성 + 문서와 연결
            await session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.page = $page, c.chunk_index = $idx
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                id=chunk_id,
                text=chunk.text[:1000],  # Neo4j 저장 용량 고려
                page=chunk.page or 0,
                idx=chunk.chunk_index,
                doc_id=doc_id,
            )

            # 엔티티 노드 생성 + 청크와 연결
            for entity in extraction.entities:
                await session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.type = $type, e.category = $category
                    ON MATCH SET
                      e.type     = CASE WHEN e.type     IS NULL THEN $type     ELSE e.type     END,
                      e.category = CASE WHEN e.category IS NULL THEN $category ELSE e.category END
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    name=entity.name,
                    type=entity.entity_type,
                    category=entity.category or "",
                    chunk_id=chunk_id,
                )

            # LLM이 추출한 명시적 관계 저장
            # MERGE로 엔드포인트 엔티티도 없으면 자동 생성
            for relation in extraction.relations:
                await session.run(
                    """
                    MERGE (s:Entity {name: $source})
                    ON CREATE SET s.type = 'CONCEPT'
                    MERGE (t:Entity {name: $target})
                    ON CREATE SET t.type = 'CONCEPT'
                    MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
                    """,
                    source=relation.source,
                    target=relation.target,
                    rel_type=relation.relation_type,
                )

            # 같은 청크에 함께 등장한 엔티티 간 co-occurrence 관계 자동 생성
            entity_names = [e.name for e in extraction.entities]
            if len(entity_names) >= 2:
                await session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e1:Entity)
                    MATCH (c)-[:MENTIONS]->(e2:Entity)
                    WHERE e1.name < e2.name
                    MERGE (e1)-[r:CO_OCCURS]->(e2)
                    ON CREATE SET r.count = 1
                    ON MATCH SET r.count = r.count + 1
                    """,
                    chunk_id=chunk_id,
                )

        logger.info(f"Neo4j indexed: doc={doc_id}, chunks={len(chunks)}")


# 관계 유형별 신뢰도 가중치
# RELATED_TO: LLM이 의미를 판단한 명시적 관계 → 높은 신뢰도
# CO_OCCURS:  같은 청크에 함께 등장한 통계적 연결 → 낮은 신뢰도
_REL_WEIGHTS: dict[str, float] = {
    "RELATED_TO": 0.8,
    "CO_OCCURS":  0.3,
}
_DEFAULT_REL_WEIGHT = 0.5  # 알 수 없는 관계 유형 기본값


def _entity_specificity(degree: int) -> float:
    """
    엔티티 중요도 역수 (TF-IDF의 IDF 개념)
    degree가 클수록 (허브 엔티티) 점수를 낮춤.
    공식: 1 / log(1 + degree),  최대 1.0으로 cap
    """
    d = max(degree, 1)
    return min(1.0, 1.0 / math.log1p(d))


async def search_by_entities(entity_names: list[str], top_k: int = 5, category_hint: str = "") -> list[dict]:
    """
    엔티티명 리스트로 그래프 탐색 → 관련 청크 반환
    - 1-hop: 엔티티를 직접 MENTIONS하는 청크
             score = specificity(degree) × category_boost (힌트 일치 시 ×1.5)
    - 2-hop: RELATED_TO/CO_OCCURS로 연결된 인접 엔티티를 MENTIONS하는 청크
             score = 0.5 × rel_weight × specificity(degree)
    category_hint: 쿼리에서 감지한 의미 카테고리 (예: "Policy", "Strategy")
    """
    if not entity_names:
        return []

    driver = get_driver()
    async with driver.session() as session:
        # 1-hop: 매칭 엔티티를 직접 언급하는 청크 + 엔티티 degree 반환
        # Document(문서) 또는 DBTable(정형DB) 모두 처리
        result_1hop = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(name)
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            OPTIONAL MATCH (d:Document)-[:CONTAINS]->(c)
            OPTIONAL MATCH (t:DBTable)-[:CONTAINS]->(c)
            WITH c, d, t, e,
                 size([(e)-[:RELATED_TO|CO_OCCURS]-() | 1]) AS degree
            WHERE d IS NOT NULL OR t IS NOT NULL
            RETURN c.id AS chunk_id,
                   c.text AS text,
                   c.page AS page,
                   c.chunk_index AS chunk_index,
                   COALESCE(d.id, t.source_id) AS doc_id,
                   COALESCE(d.filename, t.name) AS filename,
                   1.0 AS hop_score,
                   degree,
                   e.category AS entity_category
            LIMIT $limit
            """,
            names=entity_names,
            limit=top_k * 3,
        )
        records_1hop = await result_1hop.data()

        # 2-hop: 매칭 엔티티와 연결된 인접 엔티티를 언급하는 청크
        # Document(문서) 또는 DBTable(정형DB) 모두 처리
        result_2hop = await session.run(
            """
            UNWIND $names AS name
            MATCH (e1:Entity)
            WHERE toLower(e1.name) CONTAINS toLower(name)
            MATCH (e1)-[r:RELATED_TO|CO_OCCURS]-(e2:Entity)
            WHERE e2.name <> e1.name
            MATCH (c:Chunk)-[:MENTIONS]->(e2)
            OPTIONAL MATCH (d:Document)-[:CONTAINS]->(c)
            OPTIONAL MATCH (t:DBTable)-[:CONTAINS]->(c)
            WITH c, d, t, r, e2,
                 size([(e2)-[:RELATED_TO|CO_OCCURS]-() | 1]) AS degree
            WHERE d IS NOT NULL OR t IS NOT NULL
            RETURN c.id AS chunk_id,
                   c.text AS text,
                   c.page AS page,
                   c.chunk_index AS chunk_index,
                   COALESCE(d.id, t.source_id) AS doc_id,
                   COALESCE(d.filename, t.name) AS filename,
                   type(r) AS rel_type,
                   degree
            LIMIT $limit
            """,
            names=entity_names,
            limit=top_k * 5,
        )
        records_2hop = await result_2hop.data()

        # DB 경유: Entity → CONCEPTUAL_LINK → DBColumn → DBTable → Chunk
        # DB 청크가 직접 MENTIONS를 갖지 않아도 컬럼-엔티티 연결로 도달
        result_db = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(name)
            MATCH (e)<-[:CONCEPTUAL_LINK]-(col:DBColumn)<-[:HAS_COLUMN]-(t:DBTable)-[:CONTAINS]->(c:Chunk)
            RETURN c.id AS chunk_id,
                   c.text AS text,
                   c.page AS page,
                   c.chunk_index AS chunk_index,
                   t.source_id AS doc_id,
                   t.name AS filename,
                   0.7 AS hop_score,
                   1 AS degree,
                   e.category AS entity_category
            LIMIT $limit
            """,
            names=entity_names,
            limit=top_k * 3,
        )
        records_db = await result_db.data()

    # 1-hop 점수 = specificity × category_boost (힌트 카테고리 일치 시 ×1.5)
    for r in records_1hop:
        boost = 1.5 if (category_hint and r.get("entity_category") == category_hint) else 1.0
        r["hop_score"] = _entity_specificity(r.get("degree") or 0) * boost

    # 2-hop 점수 = 0.5 × rel_weight × specificity
    for r in records_2hop:
        rel_weight = _REL_WEIGHTS.get(r.get("rel_type", ""), _DEFAULT_REL_WEIGHT)
        r["hop_score"] = 0.5 * rel_weight * _entity_specificity(r.get("degree") or 0)

    # DB 경유 점수 = 0.7 × category_boost
    for r in records_db:
        boost = 1.5 if (category_hint and r.get("entity_category") == category_hint) else 1.0
        r["hop_score"] = 0.7 * boost

    # 중복 chunk_id는 높은 점수 유지
    score_map: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for r in records_1hop + records_2hop + records_db:
        cid = r["chunk_id"]
        existing_score = score_map.get(cid, 0.0)
        if r["hop_score"] > existing_score:
            score_map[cid] = r["hop_score"]
            chunk_map[cid] = r

    sorted_ids = sorted(score_map, key=lambda x: score_map[x], reverse=True)
    chunks = []
    for cid in sorted_ids[:top_k]:
        r = chunk_map[cid]
        chunks.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "page": r["page"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "score": score_map[cid],
        })
    return chunks


async def get_adjacent_entities(entity_names: list[str], max_entities: int = 5) -> list[str]:
    """
    쿼리 확장용: 매칭 엔티티의 1-hop RELATED_TO 인접 엔티티 이름 반환.
    CO_OCCURS는 신뢰도가 낮으므로 RELATED_TO만 사용.
    """
    if not entity_names:
        return []

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(name)
            MATCH (e)-[:RELATED_TO]-(neighbor:Entity)
            WHERE neighbor.name <> e.name
            RETURN DISTINCT neighbor.name AS name
            LIMIT $limit
            """,
            names=entity_names,
            limit=max_entities * 3,
        )
        records = await result.data()

    existing_lower = {n.lower() for n in entity_names}
    neighbors: list[str] = []
    for r in records:
        name = r["name"]
        if name.lower() not in existing_lower and len(neighbors) < max_entities:
            neighbors.append(name)
            existing_lower.add(name.lower())
    return neighbors


async def search_shortest_path_chunks(entity_names: list[str], top_k: int = 5) -> list[dict]:
    """
    엔티티 쌍 간 최단 경로(최대 4-hop) 상의 노드를 MENTIONS하는 청크 반환.
    쿼리에서 2개 이상 엔티티가 추출됐을 때 교차 검색에 활용.
    경로가 없거나 두 엔티티가 동일하면 빈 리스트 반환.
    """
    if len(entity_names) < 2:
        return []

    driver = get_driver()
    chunks_map: dict[str, dict] = {}

    # 처음 3개 엔티티로 페어 조합 (최대 3쌍)
    names_to_use = entity_names[:3]
    pairs = [
        (names_to_use[i], names_to_use[j])
        for i in range(len(names_to_use))
        for j in range(i + 1, len(names_to_use))
    ]

    async with driver.session() as session:
        for name1, name2 in pairs:
            try:
                result = await session.run(
                    """
                    MATCH (e1:Entity), (e2:Entity)
                    WHERE toLower(e1.name) CONTAINS toLower($name1)
                      AND toLower(e2.name) CONTAINS toLower($name2)
                      AND e1 <> e2
                    MATCH p = shortestPath((e1)-[:RELATED_TO|CO_OCCURS*..4]-(e2))
                    WITH nodes(p) AS path_nodes, length(p) AS path_len
                    WHERE path_len > 0
                    UNWIND path_nodes AS e
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(c)
                    OPTIONAL MATCH (t:DBTable)-[:CONTAINS]->(c)
                    WITH c, d, t
                    WHERE d IS NOT NULL OR t IS NOT NULL
                    RETURN DISTINCT c.id AS chunk_id,
                           c.text AS text,
                           c.page AS page,
                           c.chunk_index AS chunk_index,
                           COALESCE(d.id, t.source_id) AS doc_id,
                           COALESCE(d.filename, t.name) AS filename,
                           0.7 AS hop_score
                    LIMIT $limit
                    """,
                    name1=name1,
                    name2=name2,
                    limit=top_k * 2,
                )
                records = await result.data()
                for r in records:
                    cid = r["chunk_id"]
                    if cid not in chunks_map:
                        chunks_map[cid] = r
                if records:
                    logger.debug(f"[shortest_path] {name1}↔{name2}: {len(records)}개 청크")
            except Exception as e:
                logger.debug(f"[shortest_path] {name1}↔{name2} 경로 없음: {e}")

    return [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "page": r["page"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "score": r["hop_score"],
        }
        for r in list(chunks_map.values())[:top_k]
    ]


async def get_matched_entity_names(entity_names: list[str]) -> list[str]:
    """
    CONTAINS 매칭으로 Neo4j에 실제 저장된 엔티티 이름 반환.
    빔 서치의 시작점 확인용 (이후 탐색은 exact-match 사용).
    """
    if not entity_names:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(name)
            RETURN DISTINCT e.name AS name
            LIMIT 20
            """,
            names=entity_names,
        )
        records = await result.data()
    return [r["name"] for r in records]


async def get_related_to_neighbors(
    entity_names: list[str],
    exclude: set[str] | None = None,
    max_per_entity: int = 10,
) -> list[str]:
    """
    RELATED_TO 1-hop 이웃 이름 반환.
    빔 서치의 각 hop에서 후보 엔티티를 수집할 때 사용.
    exclude: 이미 방문한 엔티티 (재방문 방지)
    """
    if not entity_names:
        return []
    exclude_list = list(exclude) if exclude else []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name})-[:RELATED_TO]-(neighbor:Entity)
            WHERE NOT neighbor.name IN $exclude
            RETURN DISTINCT neighbor.name AS name
            LIMIT $limit
            """,
            names=entity_names,
            exclude=exclude_list,
            limit=len(entity_names) * max_per_entity,
        )
        records = await result.data()
    return [r["name"] for r in records]


async def get_chunks_by_exact_entities(entity_names: list[str], top_k: int = 5) -> list[dict]:
    """
    정확한 엔티티 이름 매칭으로 청크 반환.
    빔 서치에서 선택된 엔티티의 청크를 수집할 때 사용.
    """
    if not entity_names:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name})
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            OPTIONAL MATCH (d:Document)-[:CONTAINS]->(c)
            OPTIONAL MATCH (t:DBTable)-[:CONTAINS]->(c)
            WITH c, d, t
            WHERE d IS NOT NULL OR t IS NOT NULL
            RETURN DISTINCT c.id AS chunk_id,
                   c.text AS text,
                   c.page AS page,
                   c.chunk_index AS chunk_index,
                   COALESCE(d.id, t.source_id) AS doc_id,
                   COALESCE(d.filename, t.name) AS filename
            LIMIT $limit
            """,
            names=entity_names,
            limit=top_k * 2,
        )
        records = await result.data()
    return [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "page": r["page"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
        }
        for r in records[:top_k]
    ]


async def get_graph_data(limit: int = 100) -> dict:
    """그래프 시각화용 노드/엣지 데이터 반환
    - 문서: Entity 노드 + RELATED_TO/CO_OCCURS 엣지
    - 정형DB: DBTable/DBColumn 노드 + HAS_COLUMN/REFERENCES 엣지
    """
    driver = get_driver()
    async with driver.session() as session:
        # ── 엔티티 노드/엣지 ───────────────────────────────────────────────────
        entity_node_result = await session.run(
            """
            MATCH (e:Entity)
            WHERE (e)-[:RELATED_TO]-() OR (e)-[:CO_OCCURS]-() OR ()-[:MENTIONS]->(e)
            RETURN e.name AS name, e.type AS type
            LIMIT $limit
            """,
            limit=limit,
        )
        entity_nodes_raw = await entity_node_result.data()

        entity_edge_result = await session.run(
            """
            MATCH (s:Entity)-[r:RELATED_TO|CO_OCCURS]->(t:Entity)
            RETURN s.name AS source, t.name AS target,
                   COALESCE(r.type, type(r)) AS type
            LIMIT $limit
            """,
            limit=limit,
        )
        entity_edges_raw = await entity_edge_result.data()

        # ── DBTable/DBColumn 노드/엣지 ─────────────────────────────────────────
        pg_node_result = await session.run(
            """
            MATCH (t:DBTable)
            RETURN t.name AS name
            LIMIT $limit
            """,
            limit=limit,
        )
        pg_tables_raw = await pg_node_result.data()

        pg_col_result = await session.run(
            """
            MATCH (t:DBTable)-[:HAS_COLUMN]->(col:DBColumn)
            RETURN t.name AS table_name, col.name AS col_name, col.data_type AS data_type
            LIMIT $limit
            """,
            limit=limit,
        )
        pg_cols_raw = await pg_col_result.data()

        pg_fk_result = await session.run(
            """
            MATCH (src:DBTable)-[:REFERENCES]->(dst:DBTable)
            RETURN src.name AS src_table, dst.name AS dst_table
            LIMIT $limit
            """,
            limit=limit,
        )
        pg_fks_raw = await pg_fk_result.data()

        # DBColumn 간 CONCEPTUAL_LINK
        col_clink_result = await session.run(
            """
            MATCH (c1:DBColumn)-[:CONCEPTUAL_LINK]->(c2:DBColumn)
            RETURN c1.id AS id1, c1.table_name AS tn1, c1.name AS n1,
                   c2.id AS id2, c2.table_name AS tn2, c2.name AS n2
            LIMIT $limit
            """,
            limit=limit,
        )
        col_clinks_raw = await col_clink_result.data()

        # Entity ↔ DBColumn 크로스 CONCEPTUAL_LINK
        cross_clink_result = await session.run(
            """
            MATCH (col:DBColumn)-[:CONCEPTUAL_LINK]->(e:Entity)
            RETURN col.id AS col_id, col.table_name AS col_tn, col.name AS col_name,
                   e.name AS entity_name
            LIMIT $limit
            """,
            limit=limit,
        )
        cross_clinks_raw = await cross_clink_result.data()

    # 엔티티 노드
    edge_nodes = {e["source"] for e in entity_edges_raw} | {e["target"] for e in entity_edges_raw}
    all_entity_names = {r["name"] for r in entity_nodes_raw} | edge_nodes
    type_map = {r["name"]: r["type"] for r in entity_nodes_raw}
    nodes: list[dict] = [
        {"id": name, "label": name, "type": type_map.get(name, "CONCEPT")}
        for name in all_entity_names
    ]
    edges: list[dict] = [
        {"source": r["source"], "target": r["target"], "type": r["type"]}
        for r in entity_edges_raw
    ]

    # DBTable/DBColumn 노드
    table_ids = set()
    for r in pg_tables_raw:
        nid = f"table:{r['name']}"
        if nid not in table_ids:
            nodes.append({"id": nid, "label": r["name"], "type": "DBTABLE"})
            table_ids.add(nid)

    for r in pg_cols_raw:
        col_id = f"col:{r['table_name']}.{r['col_name']}"
        nodes.append({"id": col_id, "label": r["col_name"], "type": "DBCOLUMN"})
        edges.append({"source": f"table:{r['table_name']}", "target": col_id, "type": "HAS_COLUMN"})

    for r in pg_fks_raw:
        src_id = f"table:{r['src_table']}"
        dst_id = f"table:{r['dst_table']}"
        if dst_id not in table_ids:
            nodes.append({"id": dst_id, "label": r["dst_table"], "type": "DBTABLE"})
            table_ids.add(dst_id)
        edges.append({"source": src_id, "target": dst_id, "type": "REFERENCES"})

    # DBColumn 간 CONCEPTUAL_LINK
    for r in col_clinks_raw:
        edges.append({
            "source": f"col:{r['tn1']}.{r['n1']}",
            "target": f"col:{r['tn2']}.{r['n2']}",
            "type": "CONCEPTUAL_LINK",
        })

    # Entity ↔ DBColumn 크로스 CONCEPTUAL_LINK
    entity_ids = {n["id"] for n in nodes if n.get("type") not in ("DBTABLE", "DBCOLUMN")}
    col_ids_set = {n["id"] for n in nodes if n.get("type") == "DBCOLUMN"}
    for r in cross_clinks_raw:
        col_node_id = f"col:{r['col_tn']}.{r['col_name']}"
        entity_id = r["entity_name"]
        if col_node_id in col_ids_set and entity_id in entity_ids:
            edges.append({"source": col_node_id, "target": entity_id, "type": "CONCEPTUAL_LINK"})

    return {"nodes": nodes, "edges": edges}


async def get_entity_neighbors(entity_name: str) -> dict:
    """특정 엔티티의 1-hop 이웃 노드 반환 (하위 호환용)"""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)-[r:RELATED_TO|CO_OCCURS]-(neighbor:Entity)
            WHERE toLower(e.name) = toLower($name)
            RETURN neighbor.name AS name, neighbor.type AS type,
                   type(r) AS rel_type,
                   COALESCE(r.type, type(r)) AS rel_label
            LIMIT 30
            """,
            name=entity_name,
        )
        records = await result.data()

    return {
        "center": entity_name,
        "neighbors": [
            {"name": r["name"], "type": r["type"], "relation": r["rel_label"]}
            for r in records
        ],
    }


async def get_subgraph(entity_name: str, depth: int = 2, max_nodes: int = 100) -> dict:
    """
    N-hop 서브그래프 반환
    - 중심 엔티티로부터 depth 홉 이내의 모든 노드·엣지를 포함
    - 중간 노드 간 엣지도 포함 (연결된 전체 부분 그래프)

    주의: Neo4j는 variable-length path 홉 수에 파라미터($depth)를 지원하지 않음.
          depth 값을 f-string으로 쿼리에 직접 삽입. (1~4 정수로 검증 완료)
    """
    depth = max(1, min(int(depth), 4))   # 1~4 hop 제한 (성능 + 인젝션 방지)
    max_nodes = min(int(max_nodes), 200)

    driver = get_driver()
    async with driver.session() as session:
        # Step 1: depth 홉 이내 모든 노드 수집
        # *0..N — 0홉(자기 자신) 포함, N홉까지 확장
        node_result = await session.run(
            f"""
            MATCH (center:Entity)
            WHERE toLower(center.name) = toLower($name)
            MATCH (center)-[:RELATED_TO|CO_OCCURS*0..{depth}]-(n:Entity)
            RETURN DISTINCT n.name AS name, n.type AS type
            LIMIT $max_nodes
            """,
            name=entity_name,
            max_nodes=max_nodes,
        )
        nodes_raw = await node_result.data()

        if not nodes_raw:
            return {"nodes": [], "edges": [], "center": entity_name}

        node_names = [r["name"] for r in nodes_raw]

        # Step 2: 수집된 노드들 사이의 모든 엣지 조회
        # → 중간 노드 간 연결도 포함되어 체인 형태로 시각화됨
        edge_result = await session.run(
            """
            UNWIND $node_names AS sname
            MATCH (s:Entity {name: sname})-[r:RELATED_TO|CO_OCCURS]->(t:Entity)
            WHERE t.name IN $node_names
            RETURN s.name AS source, t.name AS target,
                   type(r) AS rel_type,
                   COALESCE(r.type, type(r)) AS rel_label
            """,
            node_names=node_names,
        )
        edges_raw = await edge_result.data()

    nodes = [
        {"id": r["name"], "label": r["name"], "type": r["type"] or "CONCEPT"}
        for r in nodes_raw
    ]
    edges = [
        {"source": r["source"], "target": r["target"], "type": r["rel_label"]}
        for r in edges_raw
    ]
    logger.info(
        f"Subgraph [{entity_name}] depth={depth}: "
        f"{len(nodes)}개 노드, {len(edges)}개 엣지"
    )
    return {"nodes": nodes, "edges": edges, "center": entity_name}


async def get_chunks_subgraph(chunk_ids: list[str]) -> dict:
    """
    검색된 청크들에서 그래프 데이터 반환.
    - 문서 청크: (Chunk)-[:MENTIONS]->(Entity) + Entity 간 관계
    - 정형DB 청크: (DBTable)-[:HAS_COLUMN]->(DBColumn) + (DBTable)-[:REFERENCES]->(DBTable)
    """
    if not chunk_ids:
        return {"nodes": [], "edges": []}

    driver = get_driver()
    async with driver.session() as session:

        # ── 문서 청크: 엔티티 기반 그래프 ────────────────────────────────────────
        entity_result = await session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN $chunk_ids
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            chunk_ids=chunk_ids,
        )
        entity_nodes_raw = await entity_result.data()

        entity_edges_raw = []
        if entity_nodes_raw:
            node_names = [r["name"] for r in entity_nodes_raw]
            edge_result = await session.run(
                """
                UNWIND $node_names AS sname
                MATCH (s:Entity {name: sname})-[r:RELATED_TO|CO_OCCURS]->(t:Entity)
                WHERE t.name IN $node_names
                RETURN s.name AS source, t.name AS target,
                       type(r) AS rel_type,
                       COALESCE(r.type, type(r)) AS rel_label
                """,
                node_names=node_names,
            )
            entity_edges_raw = await edge_result.data()

        # ── 정형DB 청크: DBTable 스키마 구조 그래프 ──────────────────────────────
        pg_table_result = await session.run(
            """
            MATCH (t:DBTable)-[:CONTAINS]->(c:Chunk)
            WHERE c.id IN $chunk_ids
            RETURN DISTINCT t.name AS table_name
            """,
            chunk_ids=chunk_ids,
        )
        pg_tables_raw = await pg_table_result.data()

        pg_nodes: list[dict] = []
        pg_edges: list[dict] = []

        if pg_tables_raw:
            table_names = [r["table_name"] for r in pg_tables_raw]

            # DBTable 노드
            for tname in table_names:
                pg_nodes.append({"id": f"table:{tname}", "label": tname, "type": "DBTABLE"})

            # DBColumn 노드 + HAS_COLUMN 엣지
            col_result = await session.run(
                """
                UNWIND $table_names AS tname
                MATCH (t:DBTable {name: tname})-[:HAS_COLUMN]->(col:DBColumn)
                RETURN tname AS table_name, col.name AS col_name, col.data_type AS data_type
                """,
                table_names=table_names,
            )
            for r in await col_result.data():
                col_id = f"col:{r['table_name']}.{r['col_name']}"
                pg_nodes.append({"id": col_id, "label": r["col_name"], "type": "DBCOLUMN"})
                pg_edges.append({
                    "source": f"table:{r['table_name']}",
                    "target": col_id,
                    "type": "HAS_COLUMN",
                })

            # FK REFERENCES 엣지
            fk_result = await session.run(
                """
                UNWIND $table_names AS tname
                MATCH (src:DBTable {name: tname})-[r:REFERENCES]->(dst:DBTable)
                RETURN src.name AS src_table, dst.name AS dst_table
                """,
                table_names=table_names,
            )
            for r in await fk_result.data():
                src_id = f"table:{r['src_table']}"
                dst_id = f"table:{r['dst_table']}"
                # 참조 대상 테이블이 pg_nodes에 없으면 추가
                if not any(n["id"] == dst_id for n in pg_nodes):
                    pg_nodes.append({"id": dst_id, "label": r["dst_table"], "type": "DBTABLE"})
                pg_edges.append({"source": src_id, "target": dst_id, "type": "REFERENCES"})

        # ── 크로스-소스 연결: Entity ↔ DBColumn CONCEPTUAL_LINK ──────────────────
        cross_edges: list[dict] = []
        if entity_nodes_raw and pg_nodes:
            entity_names = [r["name"] for r in entity_nodes_raw]
            db_col_nodes = [n for n in pg_nodes if n["type"] == "DBCOLUMN"]
            if db_col_nodes:
                cross_result = await session.run(
                    """
                    UNWIND $entity_names AS ename
                    MATCH (e:Entity {name: ename})-[:CONCEPTUAL_LINK]-(col:DBColumn)
                    RETURN e.name AS entity_name, col.id AS col_id,
                           col.name AS col_name, col.table_name AS table_name
                    """,
                    entity_names=entity_names,
                )
                visible_col_ids = {n["id"] for n in db_col_nodes}
                for r in await cross_result.data():
                    col_node_id = f"col:{r['table_name']}.{r['col_name']}"
                    if col_node_id in visible_col_ids:
                        cross_edges.append({
                            "source": r["entity_name"],
                            "target": col_node_id,
                            "type": "CONCEPTUAL_LINK",
                        })

        # DBColumn 간 CONCEPTUAL_LINK (같은 세션 내에서 조회)
        if pg_nodes:
            col_node_ids_raw = [
                n["id"].replace("col:", "") for n in pg_nodes if n["type"] == "DBCOLUMN"
            ]
            if col_node_ids_raw:
                clink_result = await session.run(
                    """
                    UNWIND $col_ids AS cid
                    MATCH (c1:DBColumn {id: cid})-[:CONCEPTUAL_LINK]-(c2:DBColumn)
                    WHERE c1.id < c2.id
                    RETURN c1.id AS id1, c1.table_name AS tn1,
                           c2.id AS id2, c2.table_name AS tn2,
                           c1.name AS n1, c2.name AS n2
                    """,
                    col_ids=col_node_ids_raw,
                )
                for r in await clink_result.data():
                    cross_edges.append({
                        "source": f"col:{r['tn1']}.{r['n1']}",
                        "target": f"col:{r['tn2']}.{r['n2']}",
                        "type": "CONCEPTUAL_LINK",
                    })

    # 통합
    nodes = (
        [{"id": r["name"], "label": r["name"], "type": r["type"] or "CONCEPT"} for r in entity_nodes_raw]
        + pg_nodes
    )
    edges = (
        [{"source": r["source"], "target": r["target"], "type": r["rel_label"]} for r in entity_edges_raw]
        + pg_edges
        + cross_edges
    )
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# 스키마 기반 인덱싱 (동적 레이블)
# ---------------------------------------------------------------------------

def _safe_label(label: str) -> str:
    """Neo4j 레이블 인젝션 방지: PascalCase 영문+숫자+언더스코어만 허용"""
    if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', label):
        raise ValueError(f"유효하지 않은 Neo4j 레이블: '{label}'")
    return label


def _safe_rel(name: str) -> str:
    """Neo4j 관계명 인젝션 방지: UPPER_SNAKE_CASE만 허용"""
    if not re.match(r'^[A-Z][A-Z0-9_]*$', name):
        raise ValueError(f"유효하지 않은 Neo4j 관계명: '{name}'")
    return name


async def index_document_with_schema(
    doc_id: str,
    filename: str,
    chunks: list[Chunk],
    schema_results: list,           # list[SchemaExtractionResult]
    schema,                         # OntologySchema
    legacy_results: list[ExtractionResult] | None = None,
):
    """
    스키마 기반 Neo4j 인덱싱.
    - Document/Chunk 노드는 기존 방식과 동일하게 생성
    - 도메인 노드(동적 레이블)와 관계를 추가로 생성
    - legacy_results가 있으면 기존 Entity 노드도 함께 생성 (병행)
    """
    from core.entity_extractor import SchemaExtractionResult

    driver = get_driver()
    async with driver.session() as session:
        # Document 노드
        await session.run(
            "MERGE (d:Document {id: $id}) SET d.filename = $filename, d.chunk_count = $count",
            id=doc_id, filename=filename, count=len(chunks),
        )

        for i, (chunk, schema_result) in enumerate(zip(chunks, schema_results)):
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"

            # Chunk 노드 + Document 연결
            await session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.page = $page, c.chunk_index = $idx
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                id=chunk_id,
                text=chunk.text[:1000],
                page=chunk.page or 0,
                idx=chunk.chunk_index,
                doc_id=doc_id,
            )

            if not isinstance(schema_result, SchemaExtractionResult):
                continue

            # ── 도메인 노드 생성 ──────────────────────────────────────────────
            for node in schema_result.nodes:
                try:
                    lbl = _safe_label(node.label)
                    id_prop = schema.id_property_for(lbl)
                    id_val = node.properties.get(id_prop)
                    if not id_val:
                        continue

                    # f-string 레이블 삽입 (검증 완료된 값만)
                    await session.run(
                        f"MERGE (n:{lbl} {{{id_prop}: $id_val}}) SET n += $props",
                        id_val=id_val,
                        props=node.properties,
                    )
                except ValueError as e:
                    logger.warning(f"[스키마 인덱싱] 노드 레이블 오류: {e}")

            # ── 도메인 관계 생성 ──────────────────────────────────────────────
            for rel in schema_result.relationships:
                try:
                    fl = _safe_label(rel.from_label)
                    tl = _safe_label(rel.to_label)
                    rn = _safe_rel(rel.rel_type)
                    await session.run(
                        f"""
                        MERGE (a:{fl} {{{rel.from_id_property}: $fv}})
                        MERGE (b:{tl} {{{rel.to_id_property}: $tv}})
                        MERGE (a)-[:{rn}]->(b)
                        """,
                        fv=rel.from_id_value,
                        tv=rel.to_id_value,
                    )
                except ValueError as e:
                    logger.warning(f"[스키마 인덱싱] 관계 오류: {e}")

            # ── Chunk → 도메인 노드 연결 ──────────────────────────────────────
            for conn in schema_result.chunk_connections:
                try:
                    tl = _safe_label(conn.to_label)
                    cr = _safe_rel(conn.rel_type)
                    await session.run(
                        f"""
                        MATCH (c:Chunk {{id: $chunk_id}})
                        MERGE (n:{tl} {{{conn.to_id_property}: $id_val}})
                        MERGE (c)-[:{cr}]->(n)
                        """,
                        chunk_id=chunk_id,
                        id_val=conn.to_id_value,
                    )
                except ValueError as e:
                    logger.warning(f"[스키마 인덱싱] Chunk 연결 오류: {e}")

            # ── 기존 Entity 병행 처리 (선택) ─────────────────────────────────
            if legacy_results and i < len(legacy_results):
                legacy = legacy_results[i]
                for entity in legacy.entities:
                    await session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET e.type = $type, e.category = $category
                        ON MATCH SET
                          e.type     = CASE WHEN e.type     IS NULL THEN $type     ELSE e.type     END,
                          e.category = CASE WHEN e.category IS NULL THEN $category ELSE e.category END
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        name=entity.name,
                        type=entity.entity_type,
                        category=entity.category or "",
                        chunk_id=chunk_id,
                    )
                for relation in legacy.relations:
                    await session.run(
                        """
                        MERGE (s:Entity {name: $source})
                        ON CREATE SET s.type = 'CONCEPT'
                        MERGE (t:Entity {name: $target})
                        ON CREATE SET t.type = 'CONCEPT'
                        MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
                        """,
                        source=relation.source,
                        target=relation.target,
                        rel_type=relation.relation_type,
                    )

        logger.info(
            f"[스키마 인덱싱] doc={doc_id}, chunks={len(chunks)}, "
            f"도메인={schema.domain_name}"
        )


async def delete_document(doc_id: str):
    """문서와 관련 청크를 Neo4j에서 삭제 (고아 엔티티는 유지)"""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)
            DETACH DELETE c
            """,
            doc_id=doc_id,
        )
        await session.run(
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            doc_id=doc_id,
        )
    logger.info(f"Neo4j deleted document: {doc_id}")


# ── PostgreSQL 연동 ────────────────────────────────────────────────────────────

async def ensure_pg_constraints():
    """DBTable / DBColumn 유니크 제약 생성"""
    driver = get_driver()
    async with driver.session() as session:
        constraints = [
            "CREATE CONSTRAINT dbtable_name IF NOT EXISTS FOR (t:DBTable) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT dbcolumn_id IF NOT EXISTS FOR (c:DBColumn) REQUIRE c.id IS UNIQUE",
        ]
        for cypher in constraints:
            try:
                await session.run(cypher)
            except Exception as e:
                logger.debug(f"PG constraint already exists or failed: {e}")


async def index_postgres_table(
    table_schema,           # TableSchema from postgres_connector
    chunks,                 # list[Chunk]
    extraction_results,     # list[ExtractionResult]
):
    """
    PostgreSQL 테이블 스키마 + Row 청크를 Neo4j에 저장.

    그래프 스키마:
      (:DBTable {name, row_count, source})-[:HAS_COLUMN]->(:DBColumn {id, name, data_type})
      (:DBTable)-[:REFERENCES {fk_column, ref_column}]->(:DBTable)
      (:DBTable)-[:CONTAINS]->(:Chunk {id, text, ...})
      (:Chunk)-[:MENTIONS]->(:Entity)
    """
    driver = get_driver()
    table_name = table_schema.table_name
    source_id = f"pg_{table_name}"

    async with driver.session() as session:
        # DBTable 노드
        await session.run(
            """
            MERGE (t:DBTable {name: $name})
            SET t.row_count = $row_count,
                t.source    = 'postgres',
                t.source_id = $source_id
            """,
            name=table_name,
            row_count=table_schema.row_count,
            source_id=source_id,
        )

        # DBColumn 노드 + HAS_COLUMN 엣지
        for col in table_schema.columns:
            col_node_id = f"{table_name}.{col.name}"
            await session.run(
                """
                MERGE (c:DBColumn {id: $id})
                SET c.name      = $col_name,
                    c.data_type = $data_type,
                    c.table_name = $table_name
                WITH c
                MATCH (t:DBTable {name: $table_name})
                MERGE (t)-[:HAS_COLUMN]->(c)
                """,
                id=col_node_id,
                col_name=col.name,
                data_type=col.data_type,
                table_name=table_name,
            )

        # REFERENCES 엣지 (FK)
        for fk in table_schema.foreign_keys:
            await session.run(
                """
                MERGE (src:DBTable {name: $src})
                MERGE (dst:DBTable {name: $dst})
                MERGE (src)-[r:REFERENCES {fk_column: $fk_col, ref_column: $ref_col}]->(dst)
                """,
                src=table_name,
                dst=fk.ref_table,
                fk_col=fk.column,
                ref_col=fk.ref_column,
            )

        # 청크 노드 + CONTAINS 엣지 + 엔티티
        for chunk, extraction in zip(chunks, extraction_results):
            chunk_id = f"pg_{table_name}_chunk_{chunk.chunk_index}"

            await session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text        = $text,
                    c.chunk_index = $idx,
                    c.page        = $page,
                    c.source      = 'postgres'
                WITH c
                MATCH (t:DBTable {name: $table_name})
                MERGE (t)-[:CONTAINS]->(c)
                """,
                id=chunk_id,
                text=chunk.text[:1000],
                idx=chunk.chunk_index,
                page=chunk.page or 0,
                table_name=table_name,
            )

            # 엔티티 노드 + MENTIONS 엣지
            for entity in extraction.entities:
                await session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.type = $type, e.category = $category
                    ON MATCH SET
                      e.type     = CASE WHEN e.type     IS NULL THEN $type     ELSE e.type     END,
                      e.category = CASE WHEN e.category IS NULL THEN $category ELSE e.category END
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    name=entity.name,
                    type=entity.entity_type,
                    category=entity.category or "",
                    chunk_id=chunk_id,
                )

            # 명시적 관계
            for relation in extraction.relations:
                await session.run(
                    """
                    MERGE (s:Entity {name: $source})
                    ON CREATE SET s.type = 'CONCEPT'
                    MERGE (t:Entity {name: $target})
                    ON CREATE SET t.type = 'CONCEPT'
                    MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
                    """,
                    source=relation.source,
                    target=relation.target,
                    rel_type=relation.relation_type,
                )

            # co-occurrence
            if len(extraction.entities) >= 2:
                await session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e1:Entity)
                    MATCH (c)-[:MENTIONS]->(e2:Entity)
                    WHERE e1.name < e2.name
                    MERGE (e1)-[r:CO_OCCURS]->(e2)
                    ON CREATE SET r.count = 1
                    ON MATCH SET r.count = r.count + 1
                    """,
                    chunk_id=chunk_id,
                )

    logger.info(
        f"Neo4j PG indexed: table={table_name}, "
        f"columns={len(table_schema.columns)}, "
        f"fks={len(table_schema.foreign_keys)}, "
        f"chunks={len(chunks)}"
    )


async def delete_postgres_table(table_name: str):
    """DBTable 노드 + 연관 DBColumn, Chunk를 Neo4j에서 삭제"""
    driver = get_driver()
    async with driver.session() as session:
        # 연관 청크 삭제
        await session.run(
            """
            MATCH (t:DBTable {name: $name})-[:CONTAINS]->(c:Chunk)
            DETACH DELETE c
            """,
            name=table_name,
        )
        # 연관 DBColumn 삭제
        await session.run(
            """
            MATCH (t:DBTable {name: $name})-[:HAS_COLUMN]->(col:DBColumn)
            DETACH DELETE col
            """,
            name=table_name,
        )
        # DBTable 자체 삭제
        await session.run(
            "MATCH (t:DBTable {name: $name}) DETACH DELETE t",
            name=table_name,
        )
        # 고아 Entity 정리: 어떤 Chunk와도 MENTIONS 관계가 없는 Entity 삭제
        await session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-()
            DETACH DELETE e
            """
        )
    logger.info(f"Neo4j deleted PG table: {table_name}")


async def create_conceptual_column_links():
    """
    서로 다른 DBTable에 속한 DBColumn 중 이름이 포함 관계인 쌍을 찾아
    CONCEPTUAL_LINK 엣지를 생성한다.

    예) 주유소_가격.휘발유_가격 ↔ 유류비_이력.기준_휘발유_가격
        → '휘발유_가격'이 '기준_휘발유_가격'에 포함되므로 연결

    조건:
    - 서로 다른 테이블의 컬럼
    - 한쪽 이름이 다른 쪽 이름에 포함 (substring)
    - 짧은 쪽 컬럼명 최소 4글자 이상 (너무 짧으면 오탐 방지)
    - 중복 생성 방지: c1.id < c2.id
    """
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c1:DBColumn), (c2:DBColumn)
            WHERE c1.table_name <> c2.table_name
              AND c1.id < c2.id
            WITH c1, c2,
                 replace(toLower(c1.name), '_', '') AS n1,
                 replace(toLower(c2.name), '_', '') AS n2
            WHERE size(n1) >= 2 AND size(n2) >= 2
              AND (n1 CONTAINS n2 OR n2 CONTAINS n1)
            MERGE (c1)-[r:CONCEPTUAL_LINK]->(c2)
            ON CREATE SET r.reason = 'column_name_similarity'
            RETURN count(r) AS created
            """
        )
        record = await result.single()
        count = record["created"] if record else 0
        logger.info(f"[Neo4j] 의미적 컬럼 연결 생성: {count}개 CONCEPTUAL_LINK")


async def create_db_entity_cross_links():
    """
    DBColumn 이름과 비슷한 Entity 노드 사이에 CONCEPTUAL_LINK 엣지를 생성한다.
    정형DB 컬럼과 문서 엔티티를 크로스-소스로 연결하여 그래프에서 함께 탐색 가능하게 함.

    조건:
    - DBColumn.name ↔ Entity.name 포함 관계 (substring)
    - 짧은 쪽 이름 최소 4글자 이상
    """
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (col:DBColumn), (e:Entity)
            WITH col, e,
                 replace(toLower(col.name), '_', '') AS col_norm,
                 toLower(e.name) AS ent_norm
            WHERE size(ent_norm) >= 2
              AND (col_norm CONTAINS ent_norm OR ent_norm CONTAINS col_norm)
            MERGE (col)-[r:CONCEPTUAL_LINK]->(e)
            ON CREATE SET r.reason = 'db_entity_cross_link'
            RETURN count(r) AS created
            """
        )
        record = await result.single()
        count = record["created"] if record else 0
        logger.info(f"[Neo4j] DB-엔티티 크로스 연결 생성: {count}개 CONCEPTUAL_LINK")


async def link_db_chunks_to_entities():
    """
    CONCEPTUAL_LINK 경로를 통해 DB 청크에 MENTIONS 연결을 생성한다.

    경로: (DBTable)-[:HAS_COLUMN]->(DBColumn)-[:CONCEPTUAL_LINK]->(Entity)
          + (DBTable)-[:CONTAINS]->(Chunk)
    결과: (Chunk)-[:MENTIONS {via='column_link'}]->(Entity)

    이렇게 하면 DB 청크가 row 실데이터를 오염시키지 않으면서도
    엔티티 기반 그래프 탐색(search_by_entities)에 포함될 수 있다.
    create_db_entity_cross_links() 호출 이후에 실행해야 한다.
    """
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (t:DBTable)-[:HAS_COLUMN]->(col:DBColumn)-[:CONCEPTUAL_LINK]->(e:Entity)
            MATCH (t)-[:CONTAINS]->(c:Chunk)
            MERGE (c)-[r:MENTIONS]->(e)
            ON CREATE SET r.via = 'column_link'
            RETURN count(r) AS created
            """
        )
        record = await result.single()
        count = record["created"] if record else 0
        logger.info(f"[Neo4j] DB 청크-엔티티 MENTIONS 연결 생성: {count}개")


async def get_db_table_columns(table_names: list[str]) -> list[dict]:
    """
    DBTable 노드의 컬럼 목록 반환.
    반환 형식: [{table_name, columns: [{name, data_type}, ...]}, ...]
    """
    if not table_names:
        return []

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $names AS tname
            MATCH (t:DBTable {name: tname})
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(col:DBColumn)
            RETURN t.name AS table_name,
                   collect({name: col.name, data_type: col.data_type}) AS columns
            ORDER BY t.name
            """,
            names=table_names,
        )
        records = await result.data()

    return [
        {
            "table_name": r["table_name"],
            "columns": [c for c in r["columns"] if c.get("name")],
        }
        for r in records
    ]


async def get_entity_paths_for_chunks(
    chunk_ids: list[str],
    max_paths: int = 8,
) -> list[dict]:
    """
    청크에서 언급된 엔티티 간 RELATED_TO 관계를 텍스트 경로로 반환.
    반환 형식: [{from_entity, relation, to_entity}, ...]
    """
    if not chunk_ids:
        return []

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e1:Entity)
            MATCH (e1)-[r:RELATED_TO]->(e2:Entity)
            WHERE e1.name <> e2.name
            RETURN DISTINCT
                e1.name AS from_entity,
                COALESCE(r.type, type(r)) AS relation,
                e2.name AS to_entity
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            limit=max_paths,
        )
        records = await result.data()

    return [
        {
            "from_entity": r["from_entity"],
            "relation": r["relation"],
            "to_entity": r["to_entity"],
        }
        for r in records
    ]


async def list_postgres_tables() -> list[dict]:
    """Neo4j에 인덱싱된 DBTable 목록 반환"""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (t:DBTable)
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(col:DBColumn)
            OPTIONAL MATCH (t)-[:CONTAINS]->(c:Chunk)
            RETURN t.name      AS table_name,
                   t.row_count AS row_count,
                   count(DISTINCT col) AS column_count,
                   count(DISTINCT c)   AS chunk_count
            ORDER BY t.name
            """
        )
        records = await result.data()
    return records
