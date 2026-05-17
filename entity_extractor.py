"""
엔티티 추출기
LLM을 이용해 텍스트에서 Named Entity와 관계를 추출.

두 가지 모드:
  - 기존 모드: 고정 8종 엔티티 타입 (폴백)
  - 스키마 기반 모드: ontology_discoverer가 발견한 도메인 스키마 사용 (Phase 2)
"""
import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """다음 텍스트에서 중요한 엔티티(개체)와 그 관계를 추출해주세요.

텍스트:
{text}

다음 JSON 형식으로만 응답하세요 (다른 설명 없이):
{{
  "entities": [
    {{"name": "엔티티명", "type": "PERSON|ORGANIZATION|LOCATION|DATE|CONCEPT|PRODUCT|LAW|EVENT", "category": "Agent|Resource|Evidence|Concept|Object|Policy|Strategy|Metric"}},
    ...
  ],
  "relations": [
    {{"source": "엔티티1", "target": "엔티티2", "type": "관계유형"}},
    ...
  ]
}}

카테고리 분류 기준:
- Agent: 행위를 수행하는 주체 (회사, 부서, 사람, 기관)
- Resource: 사용되는 자원 (예산, 장비, 연료, 인력, 노선)
- Evidence: 수치·데이터 근거 (통계, 실적, 측정값)
- Concept: 추상적 개념 (프레임워크, 방법론, 일반 개념)
- Object: 행위의 대상 (고객, 화물, 수송 대상)
- Policy: 규칙·규정·지침 (법령, 내부 규정, 컴플라이언스)
- Strategy: 실행 계획·사업 전략 (추진 전략, 실행 방안)
- Metric: 수치 목표·KPI (영업이익률, 물동량 목표, 성과 지표)

규칙:
- 고유명사, 조직명, 지명, 날짜, 핵심 개념만 추출
- 엔티티는 최대 15개
- 관계는 최대 20개
- 모호한 것은 제외
- category가 불분명하면 Concept 사용"""

# type → category 자동 매핑 (LLM 분류 실패 시 fallback)
_TYPE_TO_CATEGORY: dict[str, str] = {
    "PERSON":       "Agent",
    "ORGANIZATION": "Agent",
    "LAW":          "Policy",
    "PRODUCT":      "Resource",
    "LOCATION":     "Object",
}


@dataclass
class Entity:
    name: str
    entity_type: str
    category: str = ""


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]


async def extract_entities(text: str, api_key: str) -> ExtractionResult:
    """Solar Pro 3를 이용한 엔티티/관계 추출"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")

        # 텍스트가 너무 길면 앞부분만 사용
        truncated = text[:3000] if len(text) > 3000 else text

        response = await client.chat.completions.create(
            model="solar-pro3",
            messages=[
                {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=truncated)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        entities = [
            Entity(
                name=e["name"],
                entity_type=e.get("type", "CONCEPT"),
                category=e.get("category", "") or _TYPE_TO_CATEGORY.get(e.get("type", ""), ""),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(source=r["source"], target=r["target"], relation_type=r.get("type", "RELATED_TO"))
            for r in data.get("relations", [])
            if r.get("source") and r.get("target")
        ]
        return ExtractionResult(entities=entities, relations=relations)

    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return ExtractionResult(entities=[], relations=[])


# ---------------------------------------------------------------------------
# Phase 2: 스키마 기반 추출
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT_WITH_SCHEMA = """당신은 지식 그래프 구축을 위한 정보 추출 전문가입니다.

아래 스키마는 이 문서 도메인에 대해 사전에 설계된 온톨로지입니다.
이 스키마를 기준으로 청크 텍스트에서 노드와 관계를 추출하십시오.

[도메인 스키마]
도메인: {domain_name}
핵심 구조: {key_insight}

노드 유형:
{node_types_summary}

관계 유형:
{rel_types_summary}

[청크 ID]
{chunk_id}

[청크 텍스트]
{chunk_text}

## 추출 규칙
1. 스키마에 정의된 노드 유형만 추출하십시오 (다른 유형 금지)
2. 동일한 코드/이름은 항상 같은 노드로 취급하십시오 (중복 추출 가능, MERGE로 처리됨)
3. 관계는 텍스트에서 명시적으로 표현된 것만 추출하십시오 (추론 금지)
4. 각 노드의 id_property 값을 반드시 포함하십시오
5. 노드나 관계가 없으면 빈 배열을 반환하십시오

## 출력 형식 (JSON만, 다른 설명 없이)

{{
  "nodes": [
    {{
      "label": "노드 레이블",
      "properties": {{"id_property명": "값", "기타속성": "값"}}
    }}
  ],
  "relationships": [
    {{
      "from_label": "출발 노드 레이블",
      "from_id_property": "식별자 속성명",
      "from_id_value": "식별자 값",
      "rel_type": "관계명",
      "to_label": "도착 노드 레이블",
      "to_id_property": "식별자 속성명",
      "to_id_value": "식별자 값"
    }}
  ],
  "chunk_connections": [
    {{
      "rel_type": "DEFINES | EXEMPLIFIES | DESCRIBES",
      "to_label": "노드 레이블",
      "to_id_property": "식별자 속성명",
      "to_id_value": "식별자 값"
    }}
  ]
}}"""


@dataclass
class SchemaNode:
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class SchemaRelationship:
    from_label: str
    from_id_property: str
    from_id_value: str
    rel_type: str
    to_label: str
    to_id_property: str
    to_id_value: str


@dataclass
class SchemaChunkConnection:
    rel_type: str           # DEFINES | EXEMPLIFIES | DESCRIBES
    to_label: str
    to_id_property: str
    to_id_value: str


@dataclass
class SchemaExtractionResult:
    nodes: list[SchemaNode] = field(default_factory=list)
    relationships: list[SchemaRelationship] = field(default_factory=list)
    chunk_connections: list[SchemaChunkConnection] = field(default_factory=list)


def _build_schema_summary(schema) -> tuple[str, str]:
    """스키마를 프롬프트 삽입용 텍스트로 변환"""
    node_lines = []
    for n in schema.node_types:
        props = ", ".join(p["name"] for p in n.properties)
        node_lines.append(f"  - {n.label} ({n.label_ko}): id={n.id_property}, 속성=[{props}]")
        node_lines.append(f"    찾는 법: {n.extraction_hint}")

    rel_lines = []
    for r in schema.relationship_types:
        rel_lines.append(f"  - ({r.from_label})-[{r.name}]->({r.to_label}): {r.description}")
        rel_lines.append(f"    찾는 법: {r.extraction_hint}")

    return "\n".join(node_lines), "\n".join(rel_lines)


async def extract_entities_with_schema(
    text: str,
    chunk_id: str,
    schema,          # OntologySchema
    api_key: str,
) -> SchemaExtractionResult:
    """스키마 기반 엔티티/관계 추출 (Phase 2)"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")

        node_summary, rel_summary = _build_schema_summary(schema)
        truncated = text[:3000] if len(text) > 3000 else text

        prompt = _EXTRACTION_PROMPT_WITH_SCHEMA.format(
            domain_name=schema.domain_name,
            key_insight=schema.key_insight,
            node_types_summary=node_summary,
            rel_types_summary=rel_summary or "  (없음)",
            chunk_id=chunk_id,
            chunk_text=truncated,
        )

        response = await client.chat.completions.create(
            model="solar-pro3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content or "{}")

        # 노드 파싱 (스키마에 있는 레이블만 허용)
        allowed_labels = {n.label for n in schema.node_types}
        nodes = []
        for n in raw.get("nodes", []):
            label = n.get("label", "")
            if label not in allowed_labels:
                continue
            props = n.get("properties", {})
            if not isinstance(props, dict):
                continue
            nodes.append(SchemaNode(label=label, properties=props))

        # 관계 파싱
        allowed_rels = {r.name for r in schema.relationship_types}
        relationships = []
        for r in raw.get("relationships", []):
            rel_type = r.get("rel_type", "")
            if rel_type not in allowed_rels:
                continue
            if not all(k in r for k in ("from_label", "from_id_property", "from_id_value",
                                        "to_label", "to_id_property", "to_id_value")):
                continue
            relationships.append(SchemaRelationship(
                from_label=r["from_label"],
                from_id_property=r["from_id_property"],
                from_id_value=str(r["from_id_value"]),
                rel_type=rel_type,
                to_label=r["to_label"],
                to_id_property=r["to_id_property"],
                to_id_value=str(r["to_id_value"]),
            ))

        # Chunk 연결 파싱
        allowed_chunk_rels = {"DEFINES", "EXEMPLIFIES", "DESCRIBES"}
        chunk_connections = []
        for c in raw.get("chunk_connections", []):
            rel_type = c.get("rel_type", "")
            if rel_type not in allowed_chunk_rels:
                continue
            to_label = c.get("to_label", "")
            if to_label not in allowed_labels:
                continue
            chunk_connections.append(SchemaChunkConnection(
                rel_type=rel_type,
                to_label=to_label,
                to_id_property=c.get("to_id_property", "name"),
                to_id_value=str(c.get("to_id_value", "")),
            ))

        return SchemaExtractionResult(
            nodes=nodes,
            relationships=relationships,
            chunk_connections=chunk_connections,
        )

    except Exception as e:
        logger.warning(f"[스키마 추출] 청크 {chunk_id} 실패: {e}")
        return SchemaExtractionResult()


def extract_entities_simple(text: str) -> ExtractionResult:
    """LLM 없이 간단한 규칙 기반 추출 (fallback)"""
    # 한국어 고유명사 패턴 (간단 버전)
    patterns = {
        "DATE": r"\d{4}년\s*\d{1,2}월|\d{4}-\d{2}-\d{2}",
        "ORGANIZATION": r"[가-힣]{2,10}(?:회사|기업|그룹|협회|재단|부|처|청|원|위원회)",
        "PERSON": r"[가-힣]{2,4}(?:\s*(?:대표|회장|사장|이사|장관|의원|교수|박사|씨|님))",
    }
    entities = []
    seen = set()
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            name = match.group().strip()
            if name not in seen:
                entities.append(Entity(name=name, entity_type=entity_type))
                seen.add(name)

    return ExtractionResult(entities=entities[:15], relations=[])
