import os
import json
import time
import traceback
import importlib.util
from datetime import datetime

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ── 깔끔한 xlsx 저장 ──────────────────────────────────────
def _save_clean_xlsx(xlsx_path, rows, collection_name, experiment_id, elapsed, timestamp):
    """
    시트 구성:
      ① 요약  : 모델 정보 + Excel 수식 자동 집계
      ② 결과  : 핵심 11열, 성공/실패 색상, 헤더 freeze
    """
    # ── 색상 / 폰트 ──────────────────────────────────────
    BLUE_FILL    = PatternFill("solid", fgColor="1F497D")
    TEAL_FILL    = PatternFill("solid", fgColor="17375E")
    SUCCESS_FILL = PatternFill("solid", fgColor="E2EFDA")
    FAIL_FILL    = PatternFill("solid", fgColor="FFDDC1")
    Q_FILL       = PatternFill("solid", fgColor="F5F5F5")
    H_FONT  = Font(bold=True, color="FFFFFF", name="Arial", size=9)
    B_FONT  = Font(bold=True, name="Arial", size=9)
    N_FONT  = Font(name="Arial", size=9)
    AL_C    = Alignment(horizontal="center", vertical="center", wrap_text=True)
    AL_TL   = Alignment(horizontal="left",   vertical="top",    wrap_text=True)
    AL_TC   = Alignment(horizontal="center", vertical="top")

    n = len(rows)  # 질문 수

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # ── ① 요약 시트 ──────────────────────────────────────
    ws1 = wb.create_sheet("① 요약")

    def label_row(ws, row, label, value, label_fill=None):
        lc = ws.cell(row=row, column=1, value=label)
        vc = ws.cell(row=row, column=2, value=value)
        lc.font = B_FONT
        vc.font = N_FONT
        lc.alignment = AL_C
        vc.alignment = AL_TL
        if label_fill:
            lc.fill = label_fill
            lc.font = H_FONT

    # 모델 정보 블록
    ws1.merge_cells("A1:B1")
    title = ws1["A1"]
    title.value = "📊 평가 결과 요약"
    title.font = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    title.fill = TEAL_FILL
    title.alignment = AL_C
    ws1.row_dimensions[1].height = 28

    info = [
        ("실험 ID",       experiment_id),
        ("벡터 컬렉션",   collection_name),
        ("총 질문 수",    n),
        ("평가 일시",     timestamp),
        ("소요 시간(초)", elapsed),
    ]
    for i, (label, val) in enumerate(info, start=2):
        label_row(ws1, i, label, val)

    # 구분선
    ws1.merge_cells("A8:B8")
    ws1["A8"].value = "📈 성능 지표"
    ws1["A8"].font = H_FONT
    ws1["A8"].fill = BLUE_FILL
    ws1["A8"].alignment = AL_C
    ws1.row_dimensions[8].height = 22

    # 수식으로 자동 집계 (결과 시트 참조)
    result_sheet = "② 결과"
    metrics = [
        ("Vector 성공 수",    f"=COUNTIF('② 결과'!E2:E{n+1},\"성공\")"),
        ("Vector 성공률",     f"=COUNTIF('② 결과'!E2:E{n+1},\"성공\")/COUNTA('② 결과'!E2:E{n+1})"),
        ("Hybrid 성공 수",    f"=COUNTIF('② 결과'!H2:H{n+1},\"성공\")"),
        ("Hybrid 성공률",     f"=COUNTIF('② 결과'!H2:H{n+1},\"성공\")/COUNTA('② 결과'!H2:H{n+1})"),
        ("평균 Graph 관계 수", f"=AVERAGE('② 결과'!J2:J{n+1})"),
    ]
    for i, (label, formula) in enumerate(metrics, start=9):
        lc = ws1.cell(row=i, column=1, value=label)
        vc = ws1.cell(row=i, column=2, value=formula)
        lc.font = B_FONT
        lc.alignment = AL_C
        vc.font = N_FONT
        vc.alignment = AL_TC
        # 성공률 셀은 퍼센트 형식
        if "성공률" in label:
            vc.number_format = "0.0%"
        elif "평균" in label:
            vc.number_format = "0.00"

    # 테두리
    thin = Side(style="thin", color="CCCCCC")
    for row in ws1.iter_rows(min_row=1, max_row=13, min_col=1, max_col=2):
        for cell in row:
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws1.column_dimensions["A"].width = 20
    ws1.column_dimensions["B"].width = 30

    # ── ② 결과 시트 ──────────────────────────────────────
    ws2 = wb.create_sheet("② 결과")

    headers = [
        "번호", "분류", "질문", "정답",
        "Vector 성공", "Vector 답변", "Vector 오류",
        "Hybrid 성공", "Hybrid 답변", "Graph 수", "Hybrid 오류",
    ]
    col_widths = [6, 10, 42, 32, 10, 42, 20, 10, 42, 8, 20]
    SUCCESS_COLS = {5, 8}   # Vector 성공, Hybrid 성공 (1-based)

    # 헤더
    for col_idx, (h, w) in enumerate(zip(headers, col_widths), start=1):
        cell = ws2.cell(row=1, column=col_idx, value=h)
        cell.font      = H_FONT
        cell.fill      = BLUE_FILL
        cell.alignment = AL_C
        ws2.column_dimensions[get_column_letter(col_idx)].width = w
    ws2.row_dimensions[1].height = 30

    # 데이터
    for idx, r in enumerate(rows, start=2):
        # boolean → 문자열 변환
        v_ok = r.get("vector_success")
        h_ok = r.get("hybrid_success")
        v_str = "성공" if v_ok is True or v_ok == "성공" else "실패"
        h_str = "성공" if h_ok is True or h_ok == "성공" else "실패"

        data = [
            r.get("id", ""),
            r.get("category", ""),
            r.get("question", ""),
            r.get("ground_truth", ""),
            v_str,
            r.get("vector_answer", ""),
            r.get("vector_error", ""),
            h_str,
            r.get("hybrid_answer", ""),
            r.get("hybrid_graph_count", 0),
            r.get("hybrid_error", ""),
        ]

        for col_idx, val in enumerate(data, start=1):
            cell = ws2.cell(row=idx, column=col_idx, value=val)
            cell.font = N_FONT

            if col_idx in SUCCESS_COLS:
                cell.fill      = SUCCESS_FILL if val == "성공" else FAIL_FILL
                cell.alignment = AL_TC
            elif col_idx == 3:  # 질문 열 배경
                cell.fill      = Q_FILL
                cell.alignment = AL_TL
            elif col_idx == 10:  # Graph 수
                cell.alignment = AL_TC
            else:
                cell.alignment = AL_TL

        ws2.row_dimensions[idx].height = 15

    ws2.freeze_panes = "E2"  # 질문까지 고정, 성공/실패부터 스크롤

    wb.save(xlsx_path)


# ── 경로 설정 ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
QA_DATASET_PATH = os.path.join(EVAL_DIR, "qa_dataset.json")
HYBRID_RAG_PATH = os.path.join(BASE_DIR, "pipeline", "04_hybrid_rag.py")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── 하이브리드 모듈 동적 로드 ─────────────────────────────
def load_hybrid_module():
    spec = importlib.util.spec_from_file_location("hybrid_rag_module", HYBRID_RAG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── QA 데이터셋 로드 ──────────────────────────────────────
def load_qa_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"QA 데이터셋이 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("qa_dataset.json은 list 형태여야 합니다.")

    return data


# ── QA 데이터셋의 다양한 키 형태 지원 ─────────────────────
def extract_question(item):
    for key in ["question", "query", "질문"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_ground_truth(item):
    for key in ["ground_truth", "answer", "reference_answer", "expected_answer", "정답"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_id(item, index):
    for key in ["id", "qid", "question_id", "번호"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(index + 1)


def extract_category(item):
    for key in ["category", "type", "persona", "source", "분류"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


# ── 결과 요약용 유틸 ──────────────────────────────────────
def safe_join(items, sep=" | "):
    return sep.join([str(x) for x in items if x is not None and str(x).strip()])


def summarize_vector_docs(vector_docs):
    sources = []
    scored_sources = []

    for doc in vector_docs or []:
        source = doc.get("source", "")
        score = doc.get("score", "")
        if source:
            sources.append(source)
            scored_sources.append(f"{source} ({score})")

    return {
        "sources": safe_join(dict.fromkeys(sources).keys()),
        "scored_sources": safe_join(scored_sources),
    }


def summarize_graph_relations(graph_relations, max_preview=10):
    preview = []
    for rel in (graph_relations or [])[:max_preview]:
        from_node = rel.get("from", "")
        relation = rel.get("relation", "")
        to_node = rel.get("to", "")
        if from_node and relation and to_node:
            preview.append(f"{from_node} --[{relation}]--> {to_node}")

    return {
        "count": len(graph_relations or []),
        "preview": "\n".join(preview)
    }


def truncate_text(text, limit=1500):
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


# ── 질문 1개 평가 ────────────────────────────────────────
def evaluate_one(module, item, index):
    qid = extract_id(item, index)
    category = extract_category(item)
    question = extract_question(item)
    ground_truth = extract_ground_truth(item)

    row = {
        "id": qid,
        "category": category,
        "question": question,
        "ground_truth": ground_truth,

        "vector_success": False,
        "vector_answer": "",
        "vector_sources": "",
        "vector_scored_sources": "",
        "vector_context_preview": "",
        "vector_error": "",

        "hybrid_success": False,
        "hybrid_answer": "",
        "hybrid_sources": "",
        "hybrid_scored_sources": "",
        "hybrid_graph_count": 0,
        "hybrid_graph_preview": "",
        "hybrid_context_preview": "",
        "hybrid_error": "",
    }

    if not question:
        row["vector_error"] = "질문 없음"
        row["hybrid_error"] = "질문 없음"
        return row

    # Vector Only
    try:
        vector_result = module.vector_only_rag(question, verbose=False)
        vector_docs = vector_result.get("vector_docs", [])
        vector_summary = summarize_vector_docs(vector_docs)

        row["vector_success"] = True
        row["vector_answer"] = vector_result.get("answer", "")
        row["vector_sources"] = vector_summary["sources"]
        row["vector_scored_sources"] = vector_summary["scored_sources"]
        row["vector_context_preview"] = truncate_text(vector_result.get("context", ""))
    except Exception as e:
        row["vector_error"] = f"{type(e).__name__}: {e}"

    # Hybrid
    try:
        hybrid_result = module.hybrid_rag(question, verbose=False)
        hybrid_docs = hybrid_result.get("vector_docs", [])
        hybrid_summary = summarize_vector_docs(hybrid_docs)
        graph_summary = summarize_graph_relations(hybrid_result.get("graph_relations", []))

        row["hybrid_success"] = True
        row["hybrid_answer"] = hybrid_result.get("answer", "")
        row["hybrid_sources"] = hybrid_summary["sources"]
        row["hybrid_scored_sources"] = hybrid_summary["scored_sources"]
        row["hybrid_graph_count"] = graph_summary["count"]
        row["hybrid_graph_preview"] = graph_summary["preview"]
        row["hybrid_context_preview"] = truncate_text(hybrid_result.get("context", ""))
    except Exception as e:
        row["hybrid_error"] = f"{type(e).__name__}: {e}"

    return row


# ── 전체 평가 ─────────────────────────────────────────────
def run_evaluation():
    print("[시작] QA 데이터셋 평가")
    print(f" - QA 데이터셋: {QA_DATASET_PATH}")
    print(f" - 하이브리드 모듈: {HYBRID_RAG_PATH}")

    module = load_hybrid_module()
    dataset = load_qa_dataset(QA_DATASET_PATH)

    collection_name = getattr(module, "COLLECTION_NAME", "")
    experiment_id = getattr(module, "EXPERIMENT_ID", "")

    print(f" - 벡터 컬렉션: {collection_name}")
    print(f" - 실험 ID: {experiment_id}")
    print(f" - 총 질문 수: {len(dataset)}개")

    rows = []
    start_time = time.time()

    for idx, item in enumerate(dataset):
        question = extract_question(item)
        print(f"\n[{idx + 1}/{len(dataset)}] {question[:80]}")
        try:
            row = evaluate_one(module, item, idx)
        except Exception as e:
            row = {
                "id": extract_id(item, idx),
                "category": extract_category(item),
                "question": question,
                "ground_truth": extract_ground_truth(item),

                "vector_success": False,
                "vector_answer": "",
                "vector_sources": "",
                "vector_scored_sources": "",
                "vector_context_preview": "",
                "vector_error": f"Unhandled: {type(e).__name__}: {e}",

                "hybrid_success": False,
                "hybrid_answer": "",
                "hybrid_sources": "",
                "hybrid_scored_sources": "",
                "hybrid_graph_count": 0,
                "hybrid_graph_preview": "",
                "hybrid_context_preview": "",
                "hybrid_error": f"Unhandled: {type(e).__name__}: {e}",
            }

        print(
            f"  - Vector: {'성공' if row['vector_success'] else '실패'}"
            f" | Hybrid: {'성공' if row['hybrid_success'] else '실패'}"
            f" | Graph relations: {row['hybrid_graph_count']}"
        )
        rows.append(row)

    elapsed = round(time.time() - start_time, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.xlsx")
    json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    summary_path = os.path.join(RESULTS_DIR, f"evaluation_summary_{timestamp}.json")

    df = pd.DataFrame(rows)

    # 엑셀 저장
    _save_clean_xlsx(xlsx_path, rows, collection_name, experiment_id, elapsed, timestamp)

    # 원본 json 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary = {
        "timestamp": timestamp,
        "total_questions": len(rows),
        "vector_success_count": int(df["vector_success"].sum()) if not df.empty else 0,
        "hybrid_success_count": int(df["hybrid_success"].sum()) if not df.empty else 0,
        "avg_hybrid_graph_count": round(df["hybrid_graph_count"].mean(), 2) if not df.empty else 0,
        "collection_name": collection_name,
        "experiment_id": experiment_id,
        "elapsed_seconds": elapsed,
        "xlsx_path": xlsx_path,
        "json_path": json_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[완료] 평가 종료")
    print(f" - 총 질문 수: {summary['total_questions']}")
    print(f" - Vector 성공: {summary['vector_success_count']}")
    print(f" - Hybrid 성공: {summary['hybrid_success_count']}")
    print(f" - 평균 Graph 관계 수: {summary['avg_hybrid_graph_count']}")
    print(f" - 소요 시간: {summary['elapsed_seconds']}초")
    print(f" - 결과 xlsx: {xlsx_path}")
    print(f" - 결과 json: {json_path}")

    return summary


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print("[치명 오류] 평가 중 예외 발생")
        print(f"{type(e).__name__}: {e}")
        print(traceback.format_exc())