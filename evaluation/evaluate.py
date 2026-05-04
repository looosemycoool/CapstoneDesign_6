import os
import re
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
def _save_clean_xlsx(xlsx_path, rows, collection_name, experiment_id, elapsed, timestamp, category_stats=None):
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
    # 새 컬럼 인덱스 (1-based):
    #   E=Vector 정답, I=Hybrid 정답, L=Graph 수
    metrics = [
        ("Vector 정답 수",    f"=COUNTIF('② 결과'!E2:E{n+1},\"정답\")"),
        ("Vector 정답률",     f"=COUNTIF('② 결과'!E2:E{n+1},\"정답\")/COUNTA('② 결과'!E2:E{n+1})"),
        ("Hybrid 정답 수",    f"=COUNTIF('② 결과'!I2:I{n+1},\"정답\")"),
        ("Hybrid 정답률",     f"=COUNTIF('② 결과'!I2:I{n+1},\"정답\")/COUNTA('② 결과'!I2:I{n+1})"),
        ("평균 Graph 관계 수", f"=AVERAGE('② 결과'!L2:L{n+1})"),
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

    if category_stats:
        # 빈 행 하나
        start_row = 15

        # 섹션 타이틀
        ws1.merge_cells(f"A{start_row}:F{start_row}")
        title_cell = ws1[f"A{start_row}"]
        title_cell.value = "📊 질문 유형별 성공률"
        title_cell.font = H_FONT
        title_cell.fill = TEAL_FILL
        title_cell.alignment = AL_C
        ws1.row_dimensions[start_row].height = 22

        # 테이블 헤더
        cat_headers = ["질문 유형", "질문 수", "Vector 성공 수", "Vector 성공률", "Hybrid 성공 수", "Hybrid 성공률"]
        cat_widths  = [18, 10, 14, 14, 14, 14]
        for col_idx, (h, w) in enumerate(zip(cat_headers, cat_widths), start=1):
            cell = ws1.cell(row=start_row + 1, column=col_idx, value=h)
            cell.font = H_FONT
            cell.fill = BLUE_FILL
            cell.alignment = AL_C
            ws1.column_dimensions[get_column_letter(col_idx)].width = w

        # 유형별 데이터 행
        for row_offset, (cat, stat) in enumerate(category_stats.items(), start=2):
            r = start_row + row_offset
            data = [
                cat,
                stat["total"],
                stat["vector_success_count"],
                f"{stat['vector_success_rate']}%",
                stat["hybrid_success_count"],
                f"{stat['hybrid_success_rate']}%",
            ]
            for col_idx, val in enumerate(data, start=1):
                cell = ws1.cell(row=r, column=col_idx, value=val)
                cell.font = N_FONT
                cell.alignment = AL_TC if col_idx > 1 else AL_C
                cell.border = Border(
                    left=thin, right=thin, top=thin, bottom=thin
                )

    # ── ② 결과 시트 ──────────────────────────────────────
    ws2 = wb.create_sheet("② 결과")

    headers = [
        "번호", "분류", "질문", "정답",
        "Vector 정답", "Vector 답변", "Vector 사유", "Vector 오류",
        "Hybrid 정답", "Hybrid 답변", "Hybrid 사유", "Graph 수", "Hybrid 오류",
    ]
    col_widths = [6, 10, 42, 32, 10, 42, 30, 20, 10, 42, 30, 8, 20]
    SUCCESS_COLS = {5, 9}   # Vector 정답, Hybrid 정답 (1-based)
    GRAPH_COL = 12           # Graph 수 컬럼 인덱스

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
        # vector_success/hybrid_success 는 이제 "정답 여부" (judge LLM 결과 또는 fallback)
        v_str = "정답" if r.get("vector_success") is True else "오답"
        h_str = "정답" if r.get("hybrid_success") is True else "오답"

        data = [
            r.get("id", ""),
            r.get("category", ""),
            r.get("question", ""),
            r.get("ground_truth", ""),
            v_str,
            r.get("vector_answer", ""),
            r.get("vector_judge_reason", ""),
            r.get("vector_error", ""),
            h_str,
            r.get("hybrid_answer", ""),
            r.get("hybrid_judge_reason", ""),
            r.get("hybrid_graph_count", 0),
            r.get("hybrid_error", ""),
        ]

        for col_idx, val in enumerate(data, start=1):
            cell = ws2.cell(row=idx, column=col_idx, value=val)
            cell.font = N_FONT

            if col_idx in SUCCESS_COLS:
                cell.fill      = SUCCESS_FILL if val == "정답" else FAIL_FILL
                cell.alignment = AL_TC
            elif col_idx == 3:  # 질문 열 배경
                cell.fill      = Q_FILL
                cell.alignment = AL_TL
            elif col_idx == GRAPH_COL:  # Graph 수
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

def llm_judge(question, ground_truth, generated_answer):
    if not generated_answer:
        raise ValueError("generated_answer가 비어 있습니다.")
    if not ground_truth:
        raise ValueError("ground_truth가 비어 있습니다.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 없습니다.")

    prompt = f"""당신은 대학교 학사 공지사항 챗봇의 신뢰성을 검증하는 엄격한 AI 심사위원입니다.
            아래 [질문]에 대한 [실제 정답]과 [챗봇이 생성한 답변]을 비교하여 정확성을 평가하세요.

            [질문]
            {question}

            [실제 정답 (Ground Truth)]
            {ground_truth}

            [챗봇이 생성한 답변]
            {generated_answer}

            평가 과정 (반드시 아래 순서대로 생각하세요):
            1. Ground Truth에서 반드시 포함되어야 할 '핵심 정보'들을 추출하세요.
            2. 챗봇의 답변이 그 핵심 정보들을 모두 누락 없이 포함하고 있는지 대조하세요.
            3. 챗봇의 답변에 Ground Truth에 없는 '지어낸 거짓 정보(Hallucination)'가 섞여 있는지 확인하세요. 단순한 어조의 차이나 올바른 부연 설명은 괜찮습니다.

            마지막 줄에 반드시 "correct" 또는 "incorrect"만 단독으로 출력하세요.

            판단 기준:
            - 정답의 핵심 정보가 모두 포함되어 있고, 사실과 다른 거짓 정보가 없으면 "correct" (서술 순서나 표현 방식이 달라도 무방함)
            - 핵심 정보 중 하나라도 누락되었거나, 정답과 모순되는 명백한 거짓 정보가 포함되어 있으면 "incorrect"
            """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip().lower()
        return "correct" in result and "incorrect" not in result
    except Exception as e:
        print(f"    [Judge 오류] {e}")
        return False

# ── LLM-as-judge: 답변 정답 여부 채점 ─────────────────────
def get_judge_llm():
    """채점용 LLM (기본 Upstage solar-pro).
    환경변수 EVAL_JUDGE_MODEL 로 모델 변경 가능 (예: solar-mini).
    """
    from langchain_upstage import ChatUpstage
    model = os.getenv("EVAL_JUDGE_MODEL", "solar-pro")
    return ChatUpstage(model=model, temperature=0)


def judge_answer(question, ground_truth, predicted, judge_llm):
    """LLM 으로 답변이 ground_truth 와 의미상 일치하는지 판단.
    반환: (verdict, reason). verdict ∈ {"correct", "incorrect", "skipped", "error"}.
    """
    if not predicted or not str(predicted).strip():
        return "incorrect", "답변 없음"
    if not ground_truth or not str(ground_truth).strip():
        return "skipped", "정답(ground_truth) 미제공"

    prompt = (
        "아래 RAG 챗봇 답변이 정답과 의미상 일치하는지 판단하세요.\n"
        "사소한 표현 차이는 일치로 보고, 핵심 사실(날짜/금액/조건/대상/링크 등)이 "
        "다르거나 누락되면 불일치로 판단하세요.\n\n"
        f"질문: {question}\n"
        f"정답: {ground_truth}\n"
        f"답변: {predicted}\n\n"
        "다음 JSON 한 줄로만 답하세요:\n"
        '{"verdict": "correct" 또는 "incorrect", "reason": "한 문장"}'
    )
    try:
        raw = judge_llm.invoke(prompt).content.strip()
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                v = str(obj.get("verdict", "")).strip().lower()
                rsn = str(obj.get("reason", "")).strip()
                if v in ("correct", "incorrect"):
                    return v, rsn
            except json.JSONDecodeError:
                pass
        low = raw.lower()
        if "incorrect" in low:
            return "incorrect", raw[:200]
        if "correct" in low:
            return "correct", raw[:200]
        return "error", f"파싱 실패: {raw[:200]}"
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


# ── 질문 1개 평가 ────────────────────────────────────────
def evaluate_one(module, item, index, judge_llm=None):
    qid = extract_id(item, index)
    category = extract_category(item)
    question = extract_question(item)
    ground_truth = extract_ground_truth(item)

    row = {
        "id": qid,
        "category": category,
        "question": question,
        "ground_truth": ground_truth,

        # vector_success: True 면 정답으로 판정 (judge_llm 미설정 시 실행 성공만 의미)
        "vector_success": False,
        "vector_judge_verdict": "skipped",
        "vector_judge_reason": "",
        "vector_answer": "",
        "vector_sources": "",
        "vector_scored_sources": "",
        "vector_context_preview": "",
        "vector_error": "",

        "hybrid_success": False,
        "hybrid_judge_verdict": "skipped",
        "hybrid_judge_reason": "",
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

    # Vector Only — 실행
    vector_answer = ""
    try:
        vector_result = module.vector_only_rag(question, verbose=False)
        vector_docs = vector_result.get("vector_docs", [])
        vector_summary = summarize_vector_docs(vector_docs)

        vector_answer = vector_result.get("answer", "")
        row["vector_answer"] = vector_answer
        row["vector_sources"] = vector_summary["sources"]
        row["vector_scored_sources"] = vector_summary["scored_sources"]
        row["vector_context_preview"] = truncate_text(vector_result.get("context", ""))
    except Exception as e:
        row["vector_error"] = f"{type(e).__name__}: {e}"

    # Vector — 채점
    if row["vector_error"]:
        row["vector_judge_verdict"] = "error"
        row["vector_judge_reason"] = "파이프라인 오류"
    elif judge_llm is not None:
        verdict, reason = judge_answer(question, ground_truth, vector_answer, judge_llm)
        row["vector_judge_verdict"] = verdict
        row["vector_judge_reason"] = reason
        row["vector_success"] = (verdict == "correct")
    else:
        # judge LLM 미설정: 이전 동작 유지 (실행 성공 = success)
        row["vector_success"] = True
        row["vector_judge_reason"] = "judge 미설정"

    # Hybrid — 실행
    hybrid_answer = ""
    try:
        hybrid_result = module.hybrid_rag(question, verbose=False)
        hybrid_docs = hybrid_result.get("vector_docs", [])
        hybrid_summary = summarize_vector_docs(hybrid_docs)
        graph_summary = summarize_graph_relations(hybrid_result.get("graph_relations", []))

        hybrid_answer = hybrid_result.get("answer", "")
        row["hybrid_answer"] = hybrid_answer
        row["hybrid_success"] = llm_judge(question, ground_truth, hybrid_answer)
        row["hybrid_sources"] = hybrid_summary["sources"]
        row["hybrid_scored_sources"] = hybrid_summary["scored_sources"]
        row["hybrid_graph_count"] = graph_summary["count"]
        row["hybrid_graph_preview"] = graph_summary["preview"]
        row["hybrid_context_preview"] = truncate_text(hybrid_result.get("context", ""))
    except Exception as e:
        row["hybrid_error"] = f"{type(e).__name__}: {e}"

    # Hybrid — 채점
    if row["hybrid_error"]:
        row["hybrid_judge_verdict"] = "error"
        row["hybrid_judge_reason"] = "파이프라인 오류"
    elif judge_llm is not None:
        verdict, reason = judge_answer(question, ground_truth, hybrid_answer, judge_llm)
        row["hybrid_judge_verdict"] = verdict
        row["hybrid_judge_reason"] = reason
        row["hybrid_success"] = (verdict == "correct")
    else:
        row["hybrid_success"] = True
        row["hybrid_judge_reason"] = "judge 미설정"

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

    # LLM-as-judge: 답변이 ground_truth 와 일치하는지 채점
    try:
        judge_llm = get_judge_llm()
        judge_model = os.getenv("EVAL_JUDGE_MODEL", "solar-pro")
        print(f" - Judge LLM: {judge_model} (Upstage)")
    except Exception as e:
        judge_llm = None
        print(f" - Judge LLM 사용 불가 ({type(e).__name__}: {e})")
        print(f"   → 정답 비교 없이 진행. 'success'는 실행 성공만 의미.")

    print(f" - 벡터 컬렉션: {collection_name}")
    print(f" - 실험 ID: {experiment_id}")
    print(f" - 총 질문 수: {len(dataset)}개")

    rows = []
    start_time = time.time()

    for idx, item in enumerate(dataset):
        question = extract_question(item)
        print(f"\n[{idx + 1}/{len(dataset)}] {question[:80]}")
        try:
            row = evaluate_one(module, item, idx, judge_llm)
        except Exception as e:
            row = {
                "id": extract_id(item, idx),
                "category": extract_category(item),
                "question": question,
                "ground_truth": extract_ground_truth(item),

                "vector_success": False,
                "vector_judge_verdict": "error",
                "vector_judge_reason": "평가 핸들러 오류",
                "vector_answer": "",
                "vector_sources": "",
                "vector_scored_sources": "",
                "vector_context_preview": "",
                "vector_error": f"Unhandled: {type(e).__name__}: {e}",

                "hybrid_success": False,
                "hybrid_judge_verdict": "error",
                "hybrid_judge_reason": "평가 핸들러 오류",
                "hybrid_answer": "",
                "hybrid_sources": "",
                "hybrid_scored_sources": "",
                "hybrid_graph_count": 0,
                "hybrid_graph_preview": "",
                "hybrid_context_preview": "",
                "hybrid_error": f"Unhandled: {type(e).__name__}: {e}",
            }

        v_label = "정답" if row["vector_success"] else f"오답({row.get('vector_judge_verdict', '?')})"
        h_label = "정답" if row["hybrid_success"] else f"오답({row.get('hybrid_judge_verdict', '?')})"
        print(
            f"  - Vector: {v_label}"
            f" | Hybrid: {h_label}"
            f" | Graph relations: {row['hybrid_graph_count']}"
        )
        rows.append(row)

    elapsed = round(time.time() - start_time, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.xlsx")
    json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    summary_path = os.path.join(RESULTS_DIR, f"evaluation_summary_{timestamp}.json")

    df = pd.DataFrame(rows)

    df = pd.DataFrame(rows)

    # ── 유형별 성공률 집계 (추가) ──────────────────────────
    category_stats = {}
    if not df.empty:
        for category, group in df.groupby("category"):
            total = len(group)
            category_stats[category] = {
                "total": total,
                "vector_success_count": int(group["vector_success"].sum()),
                "vector_success_rate": round(group["vector_success"].mean() * 100, 1),
                "hybrid_success_count": int(group["hybrid_success"].sum()),
                "hybrid_success_rate": round(group["hybrid_success"].mean() * 100, 1),
            }

    # 엑셀 저장 (category_stats 추가 전달)
    _save_clean_xlsx(xlsx_path, rows, collection_name, experiment_id, elapsed, timestamp, category_stats)

    # 원본 json 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    def _count_verdict(field, target):
        if df.empty or field not in df.columns:
            return 0
        return int((df[field] == target).sum())

    summary = {
        "timestamp": timestamp,
        "total_questions": len(rows),
        "judge_used": judge_llm is not None,
        "vector_correct_count": int(df["vector_success"].sum()) if not df.empty else 0,
        "vector_incorrect_count": _count_verdict("vector_judge_verdict", "incorrect"),
        "vector_skipped_count": _count_verdict("vector_judge_verdict", "skipped"),
        "vector_error_count": _count_verdict("vector_judge_verdict", "error"),
        "hybrid_correct_count": int(df["hybrid_success"].sum()) if not df.empty else 0,
        "hybrid_incorrect_count": _count_verdict("hybrid_judge_verdict", "incorrect"),
        "hybrid_skipped_count": _count_verdict("hybrid_judge_verdict", "skipped"),
        "hybrid_error_count": _count_verdict("hybrid_judge_verdict", "error"),
        "avg_hybrid_graph_count": round(df["hybrid_graph_count"].mean(), 2) if not df.empty else 0,
        "collection_name": collection_name,
        "experiment_id": experiment_id,
        "elapsed_seconds": elapsed,
        "xlsx_path": xlsx_path,
        "json_path": json_path,
        "category_stats": category_stats,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[완료] 평가 종료")
    n = summary["total_questions"] or 1
    print(f" - 총 질문 수: {summary['total_questions']}")
    if summary["judge_used"]:
        print(
            f" - Vector 정답: {summary['vector_correct_count']}/{n} "
            f"(오답 {summary['vector_incorrect_count']}, "
            f"스킵 {summary['vector_skipped_count']}, 에러 {summary['vector_error_count']})"
        )
        print(
            f" - Hybrid 정답: {summary['hybrid_correct_count']}/{n} "
            f"(오답 {summary['hybrid_incorrect_count']}, "
            f"스킵 {summary['hybrid_skipped_count']}, 에러 {summary['hybrid_error_count']})"
        )
    else:
        print(f" - Vector 실행 성공: {summary['vector_correct_count']}")
        print(f" - Hybrid 실행 성공: {summary['hybrid_correct_count']}")
        print(f" - (judge LLM 미사용 — 정답 비교 없음)")
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