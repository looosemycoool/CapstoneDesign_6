"""
evaluate.py — RAGAS 기반 RAG 성능 평가 (v2.0)
================================================
Vector Only RAG vs Hybrid RAG를 RAGAS 4대 지표로 정량 평가합니다.

[RAGAS 지표]
  - faithfulness      : 답변이 검색 컨텍스트에 근거하는가 (환각률 역수)
  - answer_relevancy  : 답변이 질문에 얼마나 관련성이 있는가
  - context_precision : 검색된 컨텍스트 중 실제로 유용한 비율 (정밀도)
  - context_recall    : 정답 도출에 필요한 정보를 컨텍스트가 얼마나 커버하는가

[실행 전 필수]
  - .env에 OPENAI_API_KEY 설정 (RAGAS 평가 LLM으로 gpt-4o-mini 사용)
  - 04_hybrid_rag.py, Chroma DB, Neo4j가 정상 동작하는 환경

[결과 저장]
  evaluation/results/ragas_evaluation_<timestamp>.xlsx
    - ① 요약        : Vector vs Hybrid RAGAS 지표 평균 비교
    - ② 유형별 분석  : 질문 유형별 지표 분석
    - ③ Vector 상세  : 질문별 Vector RAG RAGAS 점수
    - ④ Hybrid 상세  : 질문별 Hybrid RAG RAGAS 점수
  evaluation/results/ragas_evaluation_<timestamp>.json
"""

import os
import sys
import json
import time
import traceback
import importlib.util
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH       = os.path.join(BASE_DIR, ".env")
EVAL_DIR       = os.path.join(BASE_DIR, "evaluation")
RESULTS_DIR    = os.path.join(EVAL_DIR, "results")
QA_DATASET_PATH = os.path.join(EVAL_DIR, "qa_dataset.json")
HYBRID_RAG_PATH = os.path.join(BASE_DIR, "pipeline", "04_hybrid_rag.py")

os.makedirs(RESULTS_DIR, exist_ok=True)
load_dotenv(ENV_PATH)

# ── RAGAS 평가 설정 ────────────────────────────────────────
RAGAS_LLM_MODEL        = "gpt-4o-mini"   # 평가용 LLM (답변 생성 LLM과 독립)
RAGAS_EMBEDDING_MODEL  = "text-embedding-3-small"
RAGAS_BATCH_SIZE       = 5               # RAGAS 배치 크기 (API 레이트 리밋 방지)

# ── RAGAS 임포트 ───────────────────────────────────────────
try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    print(f"[오류] RAGAS 임포트 실패: {e}")
    print("  pip install ragas datasets 를 먼저 실행하세요.")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
# 1. RAGAS 평가기 초기화
# ════════════════════════════════════════════════════════════

def build_ragas_evaluator():
    """RAGAS 평가에 사용할 LLM + Embedding 초기화"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("[오류] .env에 OPENAI_API_KEY가 없습니다. RAGAS 평가를 위해 필수입니다.")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(model=RAGAS_LLM_MODEL, api_key=openai_key, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL, api_key=openai_key)
    )

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print(f"[RAGAS] 평가기 초기화 완료")
    print(f"  - 평가 LLM      : {RAGAS_LLM_MODEL}")
    print(f"  - 평가 Embedding: {RAGAS_EMBEDDING_MODEL}")
    print(f"  - 평가 지표     : {[m.__class__.__name__ for m in metrics]}")

    return metrics


# ════════════════════════════════════════════════════════════
# 2. 하이브리드 모듈 로드
# ════════════════════════════════════════════════════════════

def load_hybrid_module():
    """04_hybrid_rag.py 동적 로드"""
    if not os.path.exists(HYBRID_RAG_PATH):
        raise FileNotFoundError(f"[오류] 파이프라인 파일 없음: {HYBRID_RAG_PATH}")
    spec   = importlib.util.spec_from_file_location("hybrid_rag_module", HYBRID_RAG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ════════════════════════════════════════════════════════════
# 3. QA 데이터셋 로드 및 필드 추출
# ════════════════════════════════════════════════════════════

def load_qa_dataset():
    if not os.path.exists(QA_DATASET_PATH):
        raise FileNotFoundError(f"[오류] QA 데이터셋 없음: {QA_DATASET_PATH}")
    with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("qa_dataset.json은 list 형태여야 합니다.")
    return data


def _get(item, *keys, default=""):
    for k in keys:
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def extract_question(item):    return _get(item, "question", "query", "질문")
def extract_ground_truth(item): return _get(item, "answer", "ground_truth", "reference_answer", "정답")
def extract_id(item, idx):     return _get(item, "id", "qid", "번호", default=str(idx + 1))
def extract_type(item):        return _get(item, "type", "category", "분류")
def extract_persona(item):     return _get(item, "persona", "source")


# ════════════════════════════════════════════════════════════
# 4. RAG 실행 (Vector Only / Hybrid)
# ════════════════════════════════════════════════════════════

def run_vector_rag(module, question: str) -> dict:
    """
    Vector Only RAG 실행
    Returns: {answer, contexts, error}
    """
    try:
        result   = module.vector_only_rag(question, verbose=False)
        answer   = result.get("answer", "")
        contexts = [doc["content"] for doc in result.get("vector_docs", []) if doc.get("content")]
        return {"answer": answer, "contexts": contexts, "error": ""}
    except Exception as e:
        return {"answer": "", "contexts": [], "error": f"{type(e).__name__}: {e}"}


def run_hybrid_rag(module, question: str) -> dict:
    """
    Hybrid RAG 실행 (Vector + Graph)
    contexts에 벡터 문서 + 그래프 관계 모두 포함
    Returns: {answer, contexts, graph_count, error}
    """
    try:
        result   = module.hybrid_rag(question, verbose=False)
        answer   = result.get("answer", "")

        # 벡터 문서 컨텍스트
        vector_contexts = [
            doc["content"]
            for doc in result.get("vector_docs", [])
            if doc.get("content")
        ]

        # 그래프 관계 → 텍스트로 변환하여 컨텍스트에 포함
        graph_relations = result.get("graph_relations", [])
        graph_contexts  = [
            f"{rel['from']} --[{rel['relation']}]--> {rel['to']}"
            for rel in graph_relations
            if rel.get("from") and rel.get("to")
        ]

        contexts    = vector_contexts + graph_contexts
        graph_count = len(graph_relations)

        return {
            "answer"     : answer,
            "contexts"   : contexts,
            "graph_count": graph_count,
            "error"      : "",
        }
    except Exception as e:
        return {"answer": "", "contexts": [], "graph_count": 0, "error": f"{type(e).__name__}: {e}"}


# ════════════════════════════════════════════════════════════
# 5. RAGAS 평가 실행
# ════════════════════════════════════════════════════════════

def run_ragas(samples_meta: list, metrics: list, label: str) -> list:
    """
    RAGAS 평가 실행
    samples_meta: [{"question", "answer", "contexts", "ground_truth", "error"}, ...]
    Returns: samples_meta에 ragas 점수 필드가 추가된 리스트
    """
    print(f"\n[RAGAS] {label} 평가 시작 ({len(samples_meta)}개 샘플)")

    # 유효한 샘플만 RAGAS에 전달 (error 없고 contexts 비어있지 않은 것)
    valid_indices = [
        i for i, s in enumerate(samples_meta)
        if not s["error"] and s["contexts"] and s["answer"]
    ]
    skip_indices  = [i for i in range(len(samples_meta)) if i not in valid_indices]

    if skip_indices:
        print(f"  [스킵] {len(skip_indices)}개 샘플 (오류 또는 컨텍스트 없음)")

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # 스킵 샘플에 None 점수 할당
    for i, s in enumerate(samples_meta):
        for m in metric_names:
            s[m] = None

    if not valid_indices:
        print(f"  [경고] 유효한 샘플이 없어 RAGAS 평가를 건너뜁니다.")
        return samples_meta

    # EvaluationDataset 구성
    ragas_samples = []
    for i in valid_indices:
        s = samples_meta[i]
        ragas_samples.append(SingleTurnSample(
            user_input        = s["question"],
            response          = s["answer"],
            retrieved_contexts= s["contexts"],
            reference         = s["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    # RAGAS evaluate 실행
    try:
        result = evaluate(
            dataset         = dataset,
            metrics         = metrics,
            raise_exceptions= False,
            show_progress   = True,
            batch_size      = RAGAS_BATCH_SIZE,
        )
        result_df = result.to_pandas()
    except Exception as e:
        print(f"  [오류] RAGAS evaluate 실패: {e}")
        return samples_meta

    # 결과를 원본 샘플에 매핑
    for result_idx, sample_idx in enumerate(valid_indices):
        if result_idx >= len(result_df):
            break
        row = result_df.iloc[result_idx]
        for m in metric_names:
            val = row.get(m, None)
            # nan → None
            if val is not None:
                try:
                    import math
                    val = None if math.isnan(float(val)) else round(float(val), 4)
                except (TypeError, ValueError):
                    val = None
            samples_meta[sample_idx][m] = val

    # 유효 샘플 점수 요약 출력
    for m in metric_names:
        scores = [s[m] for s in samples_meta if s[m] is not None]
        avg    = round(sum(scores) / len(scores), 4) if scores else None
        print(f"  {m:25s}: {avg} (평균, {len(scores)}개 유효)")

    return samples_meta


# ════════════════════════════════════════════════════════════
# 6. 전체 평가 실행
# ════════════════════════════════════════════════════════════

def run_evaluation():
    print("=" * 65)
    print("[시작] RAGAS 기반 RAG 성능 평가")
    print("=" * 65)

    # 초기화
    metrics = build_ragas_evaluator()
    module  = load_hybrid_module()
    dataset = load_qa_dataset()

    collection_name = getattr(module, "COLLECTION_NAME", "unknown")
    experiment_id   = getattr(module, "EXPERIMENT_ID",   "unknown")

    print(f"\n[설정]")
    print(f"  벡터 컬렉션 : {collection_name}")
    print(f"  실험 ID     : {experiment_id}")
    print(f"  총 질문 수  : {len(dataset)}개")

    # ── Step 1: 모든 질문에 대해 RAG 실행 ─────────────────
    print(f"\n[Step 1] RAG 응답 생성 중...")
    vector_samples = []
    hybrid_samples = []

    start_time = time.time()

    for idx, item in enumerate(dataset):
        question     = extract_question(item)
        ground_truth = extract_ground_truth(item)
        qid          = extract_id(item, idx)
        qtype        = extract_type(item)
        persona      = extract_persona(item)

        print(f"  [{idx+1:>3}/{len(dataset)}] {question[:60]}")

        if not question:
            print(f"         [스킵] 질문 없음")
            empty = {"question": "", "ground_truth": "", "answer": "",
                     "contexts": [], "error": "질문 없음",
                     "id": qid, "type": qtype, "persona": persona}
            vector_samples.append(empty)
            hybrid_samples.append({**empty, "graph_count": 0})
            continue

        # Vector RAG
        v_result = run_vector_rag(module, question)
        vector_samples.append({
            "id"          : qid,
            "type"        : qtype,
            "persona"     : persona,
            "question"    : question,
            "ground_truth": ground_truth,
            "answer"      : v_result["answer"],
            "contexts"    : v_result["contexts"],
            "error"       : v_result["error"],
        })

        # Hybrid RAG
        h_result = run_hybrid_rag(module, question)
        hybrid_samples.append({
            "id"          : qid,
            "type"        : qtype,
            "persona"     : persona,
            "question"    : question,
            "ground_truth": ground_truth,
            "answer"      : h_result["answer"],
            "contexts"    : h_result["contexts"],
            "graph_count" : h_result["graph_count"],
            "error"       : h_result["error"],
        })

        v_ok = "✓" if not v_result["error"] else "✗"
        h_ok = "✓" if not h_result["error"] else "✗"
        print(f"         Vector: {v_ok} ({len(v_result['contexts'])}개 청크)"
              f" | Hybrid: {h_ok} ({h_result.get('graph_count', 0)}개 관계)")

    rag_elapsed = round(time.time() - start_time, 2)
    print(f"\n  RAG 응답 생성 완료 ({rag_elapsed}초)")

    # ── Step 2: RAGAS 평가 ─────────────────────────────────
    print(f"\n[Step 2] RAGAS 평가 실행 중...")
    ragas_start = time.time()

    vector_samples = run_ragas(vector_samples, metrics, "Vector Only RAG")
    hybrid_samples = run_ragas(hybrid_samples, metrics, "Hybrid RAG")

    ragas_elapsed = round(time.time() - ragas_start, 2)
    print(f"\n  RAGAS 평가 완료 ({ragas_elapsed}초)")

    # ── Step 3: 결과 저장 ──────────────────────────────────
    total_elapsed = round(time.time() - start_time, 2)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path     = os.path.join(RESULTS_DIR, f"ragas_evaluation_{timestamp}.xlsx")
    json_path     = os.path.join(RESULTS_DIR, f"ragas_evaluation_{timestamp}.json")

    _save_xlsx(xlsx_path, vector_samples, hybrid_samples,
               collection_name, experiment_id, total_elapsed, timestamp)

    _save_json(json_path, vector_samples, hybrid_samples,
               collection_name, experiment_id, total_elapsed, timestamp)

    # ── 최종 요약 출력 ─────────────────────────────────────
    _print_summary(vector_samples, hybrid_samples, total_elapsed, xlsx_path)

    return {"xlsx": xlsx_path, "json": json_path}


# ════════════════════════════════════════════════════════════
# 7. 요약 계산 유틸
# ════════════════════════════════════════════════════════════

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness"     : "Faithfulness\n(환각 방지)",
    "answer_relevancy" : "Answer\nRelevancy",
    "context_precision": "Context\nPrecision",
    "context_recall"   : "Context\nRecall",
}


def _avg_metric(samples: list, metric: str) -> float | None:
    scores = [s[metric] for s in samples if s.get(metric) is not None]
    return round(sum(scores) / len(scores), 4) if scores else None


def _avg_metrics_by_type(samples: list) -> dict:
    """질문 유형별 지표 평균"""
    from collections import defaultdict
    by_type = defaultdict(list)
    for s in samples:
        by_type[s.get("type", "기타")].append(s)

    result = {}
    for qtype, type_samples in by_type.items():
        result[qtype] = {m: _avg_metric(type_samples, m) for m in METRIC_NAMES}
        result[qtype]["count"] = len(type_samples)
    return result


def _print_summary(vector_samples, hybrid_samples, elapsed, xlsx_path):
    print(f"\n{'=' * 65}")
    print("[완료] RAGAS 평가 결과 요약")
    print(f"{'=' * 65}")
    print(f"  총 소요 시간  : {elapsed}초 ({round(elapsed/60, 1)}분)")
    print(f"  총 질문 수    : {len(vector_samples)}개")
    print()
    print(f"  {'지표':25s}  {'Vector':>8}  {'Hybrid':>8}  {'차이':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")

    for m in METRIC_NAMES:
        v_avg = _avg_metric(vector_samples, m)
        h_avg = _avg_metric(hybrid_samples, m)
        diff  = round(h_avg - v_avg, 4) if (v_avg is not None and h_avg is not None) else None
        diff_str = (f"+{diff}" if diff and diff > 0 else str(diff)) if diff is not None else "N/A"
        v_str = str(v_avg) if v_avg is not None else "N/A"
        h_str = str(h_avg) if h_avg is not None else "N/A"
        print(f"  {m:25s}  {v_str:>8}  {h_str:>8}  {diff_str:>8}")

    print(f"\n  결과 파일: {xlsx_path}")


# ════════════════════════════════════════════════════════════
# 8. JSON 저장
# ════════════════════════════════════════════════════════════

def _save_json(json_path, vector_samples, hybrid_samples,
               collection_name, experiment_id, elapsed, timestamp):
    output = {
        "timestamp"      : timestamp,
        "experiment_id"  : experiment_id,
        "collection_name": collection_name,
        "elapsed_seconds": elapsed,
        "ragas_llm"      : RAGAS_LLM_MODEL,
        "summary"        : {
            "vector": {m: _avg_metric(vector_samples, m) for m in METRIC_NAMES},
            "hybrid": {m: _avg_metric(hybrid_samples, m) for m in METRIC_NAMES},
        },
        "by_type": {
            "vector": _avg_metrics_by_type(vector_samples),
            "hybrid": _avg_metrics_by_type(hybrid_samples),
        },
        "vector_details": [
            {k: v for k, v in s.items() if k != "contexts"}  # contexts는 너무 커서 제외
            for s in vector_samples
        ],
        "hybrid_details": [
            {k: v for k, v in s.items() if k != "contexts"}
            for s in hybrid_samples
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  [저장] JSON: {json_path}")


# ════════════════════════════════════════════════════════════
# 9. Excel 저장
# ════════════════════════════════════════════════════════════

# ── 색상 팔레트 ─────────────────────────────────────────────
_FILL = {
    "header_blue" : PatternFill("solid", fgColor="1F497D"),
    "header_teal" : PatternFill("solid", fgColor="17375E"),
    "header_green": PatternFill("solid", fgColor="1E5E3E"),
    "header_red"  : PatternFill("solid", fgColor="7B0000"),
    "score_high"  : PatternFill("solid", fgColor="C6EFCE"),  # ≥ 0.7
    "score_mid"   : PatternFill("solid", fgColor="FFEB9C"),  # 0.4 ~ 0.7
    "score_low"   : PatternFill("solid", fgColor="FFC7CE"),  # < 0.4
    "score_none"  : PatternFill("solid", fgColor="F2F2F2"),  # None
    "q_bg"        : PatternFill("solid", fgColor="F5F5F5"),
}
_FONT = {
    "header": Font(bold=True, color="FFFFFF", name="Arial", size=9),
    "bold"  : Font(bold=True, name="Arial", size=9),
    "normal": Font(name="Arial", size=9),
}
_AL_C  = Alignment(horizontal="center", vertical="center", wrap_text=True)
_AL_TL = Alignment(horizontal="left",   vertical="top",    wrap_text=True)
_AL_TC = Alignment(horizontal="center", vertical="top")
_THIN  = Side(style="thin", color="CCCCCC")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _score_fill(val):
    if val is None:
        return _FILL["score_none"]
    if val >= 0.7:
        return _FILL["score_high"]
    if val >= 0.4:
        return _FILL["score_mid"]
    return _FILL["score_low"]


def _set_header_row(ws, headers, fill, col_widths=None):
    ws.append(headers)
    for col_idx, cell in enumerate(ws[1], start=1):
        cell.font      = _FONT["header"]
        cell.fill      = fill
        cell.alignment = _AL_C
        cell.border    = _BORDER
    ws.row_dimensions[1].height = 36
    if col_widths:
        for i, w in enumerate(col_widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w


def _fmt_score_cell(cell, val):
    cell.value     = val if val is not None else "N/A"
    cell.fill      = _score_fill(val)
    cell.font      = _FONT["normal"]
    cell.alignment = _AL_TC
    cell.border    = _BORDER
    if isinstance(val, float):
        cell.number_format = "0.0000"


def _save_xlsx(xlsx_path, vector_samples, hybrid_samples,
               collection_name, experiment_id, elapsed, timestamp):

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # ── ① 요약 시트 ──────────────────────────────────────
    _build_summary_sheet(wb, vector_samples, hybrid_samples,
                         collection_name, experiment_id, elapsed, timestamp)

    # ── ② 유형별 분석 시트 ───────────────────────────────
    _build_type_analysis_sheet(wb, vector_samples, hybrid_samples)

    # ── ③ Vector RAG 상세 ────────────────────────────────
    _build_detail_sheet(wb, vector_samples, "③ Vector RAG 상세",
                        _FILL["header_blue"], rag_type="vector")

    # ── ④ Hybrid RAG 상세 ────────────────────────────────
    _build_detail_sheet(wb, hybrid_samples, "④ Hybrid RAG 상세",
                        _FILL["header_teal"], rag_type="hybrid")

    wb.save(xlsx_path)
    print(f"  [저장] Excel: {xlsx_path}")


def _build_summary_sheet(wb, vector_samples, hybrid_samples,
                          collection_name, experiment_id, elapsed, timestamp):
    ws = wb.create_sheet("① 요약")

    # 타이틀
    ws.merge_cells("A1:F1")
    ws["A1"].value     = "📊 RAGAS 기반 RAG 성능 평가 결과"
    ws["A1"].font      = Font(bold=True, color="FFFFFF", name="Arial", size=12)
    ws["A1"].fill      = _FILL["header_teal"]
    ws["A1"].alignment = _AL_C
    ws.row_dimensions[1].height = 32

    # 메타 정보
    meta_rows = [
        ("실험 ID",       experiment_id),
        ("벡터 컬렉션",   collection_name),
        ("총 질문 수",    str(len(vector_samples))),
        ("평가 LLM",      RAGAS_LLM_MODEL),
        ("평가 일시",     timestamp),
        ("소요 시간",     f"{elapsed}초 ({round(elapsed/60, 1)}분)"),
    ]
    for row_idx, (label, val) in enumerate(meta_rows, start=2):
        ws.cell(row=row_idx, column=1, value=label).font = _FONT["bold"]
        ws.cell(row=row_idx, column=1).alignment = _AL_C
        ws.cell(row=row_idx, column=2, value=val).font  = _FONT["normal"]
        ws.cell(row=row_idx, column=2).alignment = _AL_TL

    # 구분선
    sep_row = len(meta_rows) + 3
    ws.merge_cells(f"A{sep_row}:F{sep_row}")
    ws[f"A{sep_row}"].value     = "📈 RAGAS 지표 비교 (평균)"
    ws[f"A{sep_row}"].font      = _FONT["header"]
    ws[f"A{sep_row}"].fill      = _FILL["header_blue"]
    ws[f"A{sep_row}"].alignment = _AL_C
    ws.row_dimensions[sep_row].height = 28

    # 지표 헤더
    hdr_row = sep_row + 1
    hdrs = ["지표", "설명", "Vector RAG", "Hybrid RAG", "차이 (Hybrid-Vector)", "판정"]
    for col_idx, h in enumerate(hdrs, start=1):
        cell = ws.cell(row=hdr_row, column=col_idx, value=h)
        cell.font      = _FONT["header"]
        cell.fill      = _FILL["header_green"]
        cell.alignment = _AL_C
        cell.border    = _BORDER
    ws.row_dimensions[hdr_row].height = 28

    descriptions = {
        "faithfulness"     : "답변이 검색 컨텍스트에 근거하는가 (1.0 = 환각 없음)",
        "answer_relevancy" : "답변이 질문에 얼마나 관련성이 있는가",
        "context_precision": "검색된 컨텍스트 중 실제로 유용한 비율",
        "context_recall"   : "정답 도출에 필요한 정보를 컨텍스트가 커버하는 비율",
    }

    for m_idx, m in enumerate(METRIC_NAMES):
        data_row = hdr_row + 1 + m_idx
        v_avg = _avg_metric(vector_samples, m)
        h_avg = _avg_metric(hybrid_samples, m)
        diff  = round(h_avg - v_avg, 4) if (v_avg is not None and h_avg is not None) else None

        if diff is None:
            judgement = "N/A"
        elif diff > 0.05:
            judgement = "✅ Hybrid 우수"
        elif diff < -0.05:
            judgement = "⚠️ Vector 우수"
        else:
            judgement = "➖ 유사"

        cells_data = [
            (m, _FONT["bold"], _AL_C, None),
            (descriptions[m], _FONT["normal"], _AL_TL, None),
        ]
        for col_idx, (val, font, align, fill) in enumerate(cells_data, start=1):
            c = ws.cell(row=data_row, column=col_idx, value=val)
            c.font = font; c.alignment = align; c.border = _BORDER

        _fmt_score_cell(ws.cell(row=data_row, column=3), v_avg)
        _fmt_score_cell(ws.cell(row=data_row, column=4), h_avg)

        # 차이 셀
        diff_cell = ws.cell(row=data_row, column=5)
        diff_cell.value     = diff if diff is not None else "N/A"
        diff_cell.font      = Font(bold=True, name="Arial", size=9,
                                   color="006400" if (diff and diff > 0) else
                                         "8B0000" if (diff and diff < 0) else "000000")
        diff_cell.alignment = _AL_TC
        diff_cell.border    = _BORDER
        if isinstance(diff, float):
            diff_cell.number_format = "+0.0000;-0.0000;0.0000"

        judgement_cell = ws.cell(row=data_row, column=6, value=judgement)
        judgement_cell.font      = _FONT["normal"]
        judgement_cell.alignment = _AL_TC
        judgement_cell.border    = _BORDER
        ws.row_dimensions[data_row].height = 20

    # 범례
    legend_row = hdr_row + len(METRIC_NAMES) + 2
    ws.cell(row=legend_row, column=1, value="* 점수 범례:").font = _FONT["bold"]
    legend_data = [
        (legend_row,   2, "🟢 0.70 이상 (우수)", _FILL["score_high"]),
        (legend_row+1, 2, "🟡 0.40 ~ 0.69 (보통)", _FILL["score_mid"]),
        (legend_row+2, 2, "🔴 0.40 미만 (개선 필요)", _FILL["score_low"]),
    ]
    for row, col, txt, fill in legend_data:
        c = ws.cell(row=row, column=col, value=txt)
        c.fill = fill; c.font = _FONT["normal"]; c.alignment = _AL_TL

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 46
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 20
    ws.column_dimensions["F"].width = 18


def _build_type_analysis_sheet(wb, vector_samples, hybrid_samples):
    ws = wb.create_sheet("② 유형별 분석")

    v_by_type = _avg_metrics_by_type(vector_samples)
    h_by_type = _avg_metrics_by_type(hybrid_samples)
    all_types = sorted(set(list(v_by_type.keys()) + list(h_by_type.keys())))

    # 헤더
    hdrs = ["질문 유형", "샘플 수", "RAG 방식"] + [METRIC_LABELS.get(m, m) for m in METRIC_NAMES]
    _set_header_row(ws, hdrs, _FILL["header_green"],
                    col_widths=[22, 10, 14, 16, 16, 16, 16])

    row_idx = 2
    for qtype in all_types:
        v_data = v_by_type.get(qtype, {})
        h_data = h_by_type.get(qtype, {})
        count  = v_data.get("count", 0)

        for rag_label, data, fill in [
            ("Vector", v_data, _FILL["header_blue"]),
            ("Hybrid", h_data, _FILL["header_teal"]),
        ]:
            ws.cell(row=row_idx, column=1, value=qtype if rag_label == "Vector" else "").font = _FONT["bold"]
            ws.cell(row=row_idx, column=1).alignment = _AL_C
            ws.cell(row=row_idx, column=2, value=count if rag_label == "Vector" else "").alignment = _AL_TC
            rag_cell = ws.cell(row=row_idx, column=3, value=rag_label)
            rag_cell.fill = fill; rag_cell.font = _FONT["header"]; rag_cell.alignment = _AL_TC

            for col_offset, m in enumerate(METRIC_NAMES, start=4):
                _fmt_score_cell(ws.cell(row=row_idx, column=col_offset), data.get(m))

            ws.row_dimensions[row_idx].height = 18
            row_idx += 1

        # 구분선 행
        for col in range(1, 8):
            ws.cell(row=row_idx, column=col).border = _BORDER
        ws.row_dimensions[row_idx].height = 6
        row_idx += 1

    ws.freeze_panes = "A2"


def _build_detail_sheet(wb, samples, sheet_name, header_fill, rag_type="vector"):
    ws = wb.create_sheet(sheet_name)

    hdrs = [
        "번호", "유형", "페르소나", "질문", "정답",
        "생성 답변",
        "Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall",
        "평균 점수", "오류",
    ]
    col_widths = [6, 18, 12, 46, 36, 46, 14, 14, 14, 14, 12, 28]
    if rag_type == "hybrid":
        hdrs.insert(6, "Graph 관계 수")
        col_widths.insert(6, 12)

    _set_header_row(ws, hdrs, header_fill, col_widths=col_widths)

    SCORE_COL_START = 8 if rag_type == "hybrid" else 7  # 1-based

    for row_idx, s in enumerate(samples, start=2):
        scores = [s.get(m) for m in METRIC_NAMES]
        valid_scores = [sc for sc in scores if sc is not None]
        avg_score = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

        row_data = [
            s.get("id", ""),
            s.get("type", ""),
            s.get("persona", ""),
            s.get("question", ""),
            s.get("ground_truth", ""),
        ]
        if rag_type == "hybrid":
            row_data.append(s.get("graph_count", 0))
        row_data.append(s.get("answer", ""))

        # 일반 셀 삽입
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = _FONT["normal"]
            cell.border = _BORDER
            if col_idx == 4:  # 질문
                cell.fill = _FILL["q_bg"]
                cell.alignment = _AL_TL
            elif col_idx == 1:
                cell.alignment = _AL_TC
            else:
                cell.alignment = _AL_TL

        # RAGAS 점수 셀 (색상 포함)
        score_col = SCORE_COL_START
        for m in METRIC_NAMES:
            _fmt_score_cell(ws.cell(row=row_idx, column=score_col), s.get(m))
            score_col += 1

        # 평균 점수
        _fmt_score_cell(ws.cell(row=row_idx, column=score_col), avg_score)
        score_col += 1

        # 오류
        err_cell = ws.cell(row=row_idx, column=score_col, value=s.get("error", ""))
        err_cell.font = Font(name="Arial", size=9, color="CC0000") if s.get("error") else _FONT["normal"]
        err_cell.alignment = _AL_TL
        err_cell.border = _BORDER

        ws.row_dimensions[row_idx].height = 16

    ws.freeze_panes = "F2"


# ════════════════════════════════════════════════════════════
# 10. 메인
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        result = run_evaluation()
        print(f"\n[완료] 결과 파일:")
        print(f"  Excel : {result['xlsx']}")
        print(f"  JSON  : {result['json']}")
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 평가를 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[치명 오류] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        sys.exit(1)