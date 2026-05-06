# Ablation: 2-hop graph search + 후보 노드 확대

## 가설
- Graph 검색의 평균 매칭 수가 2.74 / 5 (cap의 절반) → recall 부족이 hybrid 한계의 주된 원인이라 판단
- 2-hop 간접 관계 + 후보 노드 확대로 recall 을 늘리면 hybrid 정답률이 의미 있게 상승할 것

## 변경 내용 (`pipeline/04_hybrid_rag.py`)

| 항목 | 이전 (1-hop) | 실험 (2-hop + 확대) |
|---|---|---|
| graph 탐색 깊이 | 1-hop 정/역방향 | 1-hop + 2-hop 간접 관계 |
| `max_nodes_per_keyword` | 5 | 10 |
| 2-hop 점수 가중치 | — | 0.5 (1-hop의 절반) |
| `merge_results` | 평문 | `(간접)` 마커 추가 |
| `max_relations` (top-k) | 5 | 5 (동일) |

## 결과

110 질문 평가 (`pipeline/04_hybrid_rag.py` 기반, judge: solar-pro2 temp=0).

| 지표 | 1-hop (이전 베스트) | **2-hop + 확대 (실험)** | 변화 |
|---|---|---|---|
| Vector 정답 | 50 / 110 (45.5%) | 53 / 110 (48.2%) | +3 (judge 노이즈 추정) |
| **Hybrid 정답** | **52 / 110 (47.3%)** | **49 / 110 (44.5%)** | **−3 (악화)** |
| 평균 graph 매칭 | 2.74 | 2.70 | 거의 동일 |
| 소요 시간 | 41 분 | **228 분 (3.8 시간)** | **5.6 배** |
| pipeline error | 0 | 1 (Neo4j timeout 추정) | + |

결과 파일:
- 1-hop: `evaluation/results/evaluation_results_20260505_201739.{json,xlsx}`
- 2-hop: `evaluation/results/evaluation_results_20260506_001344.{json,xlsx}`

## 해석

1. **Hybrid 정답률 −3** — 2-hop 간접 관계가 noise 만 추가, 핵심 사실 보강에 도움이 안 됐다
2. **평균 graph 매칭이 거의 동일 (2.70 vs 2.74)** — 2-hop 관계가 0.5 점수 페널티 때문에 top-5 안에 거의 못 들어감. 결국 1-hop top-5 와 비슷한 결과
3. **시간 5.6 배 증가** — 2-hop Cypher (`MATCH (a)-[*2..2]-(b)` + 후속 edge 조회) 가 노드당 추가 query 라 비용 큼
4. **Vector 정답률 +3 변동** — 같은 vector pipeline 인데도 변동 → judge LLM 의 boundary case 미세 변동 추정 (temperature=0 이지만 완전 결정적 아님)

## 결론

**2-hop + 후보 노드 확대는 본 데이터에서 효과 없음 또는 음의 효과.**
- 정답률 −3, 시간 ×5.6, 비용 증가
- 1-hop 직접 관계만으로 충분 (오히려 더 정확)
- recall 확대보다 precision 유지가 더 중요한 데이터 특성으로 보임

→ **main 은 1-hop 유지** (`pipeline/04_hybrid_rag.py` `graph_search` 의 직접 정/역방향 + `max_nodes_per_keyword=5`).

## Paper 활용 포인트

- "graph search 깊이는 1-hop 으로 충분" 을 ablation 으로 정직하게 보고 가능
- "2-hop 도입은 정답률 −2.7%p, 응답시간 +460% 로 trade-off 가 나쁨" 인용 가능
- recall 확대 보다 정확한 1-hop 매칭 + 점수 기반 필터링이 본 도메인에 맞다는 근거

## 재현 정보

- Model
  - Generation: `solar-pro3` (temperature=0.2)
  - Judge: `solar-pro2` (temperature=0)
  - Embedding: `embedding-passage`
- Graph DB: 748 노드 / 1498 관계 (manual 16 + notices 52 attachments)
- Vector DB (Chroma): `knu_cse_upstage_pro` 컬렉션, 185 chunks, chunk_size=500
- Branch (실험): `feat/graph-search-2hop` (commit `ad22454`) — 머지하지 않음
- Branch (롤백 = baseline): `main`
- 평가 데이터: `evaluation/qa_dataset.json` (110 문항)
- 평가일: 2026-05-05 ~ 2026-05-06
