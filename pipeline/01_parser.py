import os
import json
import zipfile
import subprocess
import shutil
import tempfile
import pandas as pd
import opendataloader_pdf
from docx import Document

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACHMENT_DIR = os.path.join(BASE_DIR, "data", "attachments")
PARSED_DIR     = os.path.join(BASE_DIR, "data", "parsed")
RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
TMP_DIR        = os.path.join(BASE_DIR, "data", "tmp_convert")
LIBREOFFICE    = r"C:\Program Files\LibreOffice\program\soffice.exe"

# ── opendataloader-pdf 옵션 ────────────────────────────────────────────────────
# 스캔 PDF / 한글 문서가 많을 경우 hybrid=True, ocr_lang="ko,en" 으로 설정
# 단, hybrid 모드를 사용하려면 먼저 별도 터미널에서 서버를 실행해야 합니다:
#   pip install "opendataloader-pdf[hybrid]"
#   opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "ko,en"
USE_HYBRID = False   # 스캔 PDF 포함 시 True
HYBRID_BACKEND = "docling-fast"   # USE_HYBRID=True 시 사용

os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


# ── PDF 파싱 (opendataloader-pdf) ──────────────────────────────────────────────

def _odl_convert_batch(pdf_paths: list[str], output_dir: str) -> None:
    """
    opendataloader_pdf.convert()는 호출마다 JVM 프로세스를 생성하므로,
    반드시 여러 파일을 한 번에 묶어서 호출해야 성능이 유지됩니다.
    """
    kwargs: dict = dict(
        input_path=pdf_paths,
        output_dir=output_dir,
        format="markdown",          # Markdown: RAG 청킹에 최적
    )
    if USE_HYBRID:
        kwargs["hybrid"] = HYBRID_BACKEND   # 복잡한 표·스캔 PDF 정확도 향상

    opendataloader_pdf.convert(**kwargs)


def parse_pdf(file_path: str) -> str:
    """
    단일 PDF → Markdown 텍스트 추출 (opendataloader-pdf 사용).

    변경 이유:
      - pdfplumber 대비 읽기 순서(XY-Cut++) 및 표 구조 보존 정확도 우수
      - 벤치마크 #1 (overall 0.90, 표 0.93)
      - 한국어 스캔 PDF는 hybrid + OCR 모드로 처리 가능
      - 단일 호출 시 JVM 오버헤드 발생 → 배치 처리 권장 (parse_all 참고)
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_out:
            _odl_convert_batch([file_path], tmp_out)
            base    = os.path.splitext(os.path.basename(file_path))[0]
            md_path = os.path.join(tmp_out, base + ".md")
            if os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as f:
                    return f.read().strip()
            print(f"  [PDF 경고] Markdown 출력 없음: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"  [PDF 오류] {file_path}: {e}")
    return ""


def parse_pdf_batch(pdf_map: dict[str, str]) -> dict[str, str]:
    """
    여러 PDF를 한 번의 JVM 호출로 파싱합니다 (권장 방식).

    Args:
        pdf_map: {원본_경로: 파일명} 딕셔너리

    Returns:
        {원본_경로: 추출된_텍스트} 딕셔너리
    """
    results: dict[str, str] = {}
    if not pdf_map:
        return results

    pdf_paths = list(pdf_map.keys())
    try:
        with tempfile.TemporaryDirectory() as tmp_out:
            _odl_convert_batch(pdf_paths, tmp_out)
            for file_path in pdf_paths:
                base    = os.path.splitext(os.path.basename(file_path))[0]
                md_path = os.path.join(tmp_out, base + ".md")
                if os.path.exists(md_path):
                    with open(md_path, encoding="utf-8") as f:
                        results[file_path] = f.read().strip()
                else:
                    print(f"  [PDF 경고] Markdown 출력 없음: {os.path.basename(file_path)}")
                    results[file_path] = ""
    except Exception as e:
        print(f"  [PDF 배치 오류] {e}")
        for p in pdf_paths:
            results.setdefault(p, "")
    return results


# ── HWP / HWPX 파싱 ───────────────────────────────────────────────────────────

def hwp_to_txt(file_path: str) -> str:
    """
    LibreOffice로 HWP/HWPX → TXT 변환.

    참고: opendataloader-pdf Roadmap에 HWP 네이티브 지원(Hancom Data Loader)이
          Q2-Q3 2026에 예정되어 있어, 추후 이 함수를 대체할 수 있습니다.
    """
    try:
        subprocess.run(
            [LIBREOFFICE, "--headless", "--convert-to", "txt:Text",
             "--outdir", TMP_DIR, file_path],
            timeout=30, capture_output=True, check=False
        )
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_path  = os.path.join(TMP_DIR, base_name + ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            os.remove(txt_path)
            return text.strip()
        print(f"  [HWP 변환 실패] txt 파일 없음: {base_name}")
    except subprocess.TimeoutExpired:
        print(f"  [HWP 타임아웃] {file_path}")
    except Exception as e:
        print(f"  [HWP 오류] {file_path}: {e}")
    return ""


# ── XLSX 파싱 ──────────────────────────────────────────────────────────────────

def parse_xlsx(file_path: str) -> str:
    """엑셀 시트의 모든 행을 파이프 구분 텍스트로 추출합니다."""
    result: list[str] = []
    try:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name).fillna("")
            result.append(f"[시트: {sheet_name}]")
            for _, row in df.iterrows():
                row_text = " | ".join(
                    str(v).strip() for v in row.values if str(v).strip()
                )
                if row_text:
                    result.append(row_text)
    except Exception as e:
        print(f"  [XLSX 오류] {file_path}: {e}")
    return "\n".join(result)


# ── DOCX 파싱 ──────────────────────────────────────────────────────────────────

def parse_docx(file_path: str) -> str:
    """DOCX 단락 및 표 텍스트를 추출합니다."""
    result: list[str] = []
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                result.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    result.append("[표] " + row_text)
    except Exception as e:
        print(f"  [DOCX 오류] {file_path}: {e}")
    return "\n".join(result)


# ── ZIP 파싱 ───────────────────────────────────────────────────────────────────

def parse_zip(file_path: str) -> str:
    """
    ZIP 압축 해제 후 내부 파일을 파싱합니다.
    내부 PDF가 여러 개일 경우 배치 변환으로 JVM 호출 횟수를 최소화합니다.
    """
    result: list[str] = []
    tmp_zip_dir = os.path.join(BASE_DIR, "data", "tmp_zip")
    try:
        os.makedirs(tmp_zip_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(tmp_zip_dir)

        # PDF는 배치로 수집 후 한꺼번에 변환
        pdf_paths: dict[str, str] = {}
        non_pdf_items: list[tuple[str, str]] = []   # (inner_path, fname)

        for root, _, files in os.walk(tmp_zip_dir):
            for fname in files:
                inner_path = os.path.join(root, fname)
                ext        = os.path.splitext(fname)[1].lower()
                if ext == ".pdf":
                    pdf_paths[inner_path] = fname
                else:
                    non_pdf_items.append((inner_path, fname))

        # PDF 배치 변환
        if pdf_paths:
            pdf_results = parse_pdf_batch(pdf_paths)
            for inner_path, text in pdf_results.items():
                fname = pdf_paths[inner_path]
                if text:
                    result.append(f"[ZIP 내부: {fname}]\n{text}")

        # 나머지 파일
        for inner_path, fname in non_pdf_items:
            ext  = os.path.splitext(fname)[1].lower()
            text = ""
            if ext in (".hwp", ".hwpx"):
                text = hwp_to_txt(inner_path)
            elif ext in (".xlsx", ".xls"):
                text = parse_xlsx(inner_path)
            elif ext == ".docx":
                text = parse_docx(inner_path)
            if text:
                result.append(f"[ZIP 내부: {fname}]\n{text}")

    except Exception as e:
        print(f"  [ZIP 오류] {file_path}: {e}")
    finally:
        shutil.rmtree(tmp_zip_dir, ignore_errors=True)

    return "\n\n".join(result)


# ── 단일 파일 라우터 ───────────────────────────────────────────────────────────

def parse_file(file_path: str) -> str:
    """파일 확장자에 따라 적절한 파서를 호출합니다."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in (".hwp", ".hwpx"):
        return hwp_to_txt(file_path)
    elif ext in (".xlsx", ".xls"):
        return parse_xlsx(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".zip":
        return parse_zip(file_path)
    else:
        print(f"  [스킵] 지원하지 않는 형식: {file_path}")
        return ""


# ── 전체 공지사항 파싱 ─────────────────────────────────────────────────────────

def parse_all() -> None:
    """
    전체 첨부파일 파싱 후 notices.json과 합쳐서 저장합니다.
    PDF는 공지 단위로 배치 변환하여 JVM 프로세스 수를 최소화합니다.
    """
    notices_path = os.path.join(RAW_DIR, "notices.json")
    with open(notices_path, encoding="utf-8") as f:
        notices = json.load(f)

    print(f"[시작] 첨부파일 파싱 (공지 {len(notices)}개)")

    for notice in notices:
        wr_id      = notice["url"].split("wr_id=")[1].split("&")[0]
        attach_dir = os.path.join(ATTACHMENT_DIR, wr_id)
        attachments = notice.get("attachments", [])
        if not attachments:
            continue

        print(f"\n[{wr_id}] {notice['title'][:40]}")

        # ① PDF 첨부파일 수집 → 배치 변환
        pdf_att_map: dict[str, dict] = {}   # {file_path: att_dict}
        for att in attachments:
            file_path = os.path.join(attach_dir, att["name"])
            if not os.path.exists(file_path):
                print(f"  [없음] {att['name']}")
                att["parsed_text"] = ""
                continue
            if os.path.splitext(att["name"])[1].lower() == ".pdf":
                pdf_att_map[file_path] = att
            else:
                print(f"  파싱 중: {att['name']}")
                att["parsed_text"] = parse_file(file_path)
                print(f"  완료: {len(att['parsed_text'])}자 추출")

        # ② PDF 배치 처리
        if pdf_att_map:
            pdf_results = parse_pdf_batch(
                {p: a["name"] for p, a in pdf_att_map.items()}
            )
            for file_path, att in pdf_att_map.items():
                text = pdf_results.get(file_path, "")
                att["parsed_text"] = text
                print(f"  완료(PDF): {att['name']} → {len(text)}자 추출")

    output_path = os.path.join(PARSED_DIR, "notices_parsed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notices, f, ensure_ascii=False, indent=2)

    total_files  = sum(len(n.get("attachments", [])) for n in notices)
    parsed_files = sum(
        1 for n in notices
        for a in n.get("attachments", [])
        if a.get("parsed_text")
    )
    print(f"\n[완료] 파싱 결과 저장 → {output_path}")
    print(f"파싱 성공: {parsed_files}/{total_files}개")


# ── 수동 업로드 파일 파싱 ──────────────────────────────────────────────────────

SUPPORTED_EXTS = {".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".docx", ".txt", ".zip"}


def parse_manual_files() -> None:
    """
    manual_files 폴더의 파일들을 파싱합니다.
    PDF는 한 번에 배치 변환합니다.
    """
    manual_dir  = os.path.join(BASE_DIR, "data", "manual_files")
    output_path = os.path.join(PARSED_DIR, "manual_parsed.json")

    all_files = [
        f for f in os.listdir(manual_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    skipped = [
        f for f in os.listdir(manual_dir)
        if os.path.splitext(f)[1].lower() not in SUPPORTED_EXTS
    ]

    for fname in skipped:
        print(f"  [스킵] {fname}")

    print(f"[시작] manual_files 파싱 ({len(all_files)}개 파일)")

    results: list[dict] = []

    # ① PDF 배치 수집
    pdf_files: dict[str, str] = {}     # {file_path: fname}
    non_pdf: list[str] = []

    for fname in all_files:
        fpath = os.path.join(manual_dir, fname)
        ext   = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            pdf_files[fpath] = fname
        else:
            non_pdf.append(fname)

    # ② PDF 배치 변환
    if pdf_files:
        print(f"  PDF 배치 변환 중 ({len(pdf_files)}개)...")
        pdf_results = parse_pdf_batch(pdf_files)
        for fpath, fname in pdf_files.items():
            text = pdf_results.get(fpath, "")
            print(f"  완료(PDF): {fname} → {len(text)}자 추출")
            results.append({"file_name": fname, "file_path": fpath, "parsed_text": text})

    # ③ 나머지 파일 개별 파싱
    for fname in non_pdf:
        fpath = os.path.join(manual_dir, fname)
        ext   = os.path.splitext(fname)[1].lower()
        print(f"  파싱 중: {fname}")
        if ext == ".txt":
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        else:
            text = parse_file(fpath)
        print(f"  완료: {len(text)}자 추출")
        results.append({"file_name": fname, "file_path": fpath, "parsed_text": text})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in results if r["parsed_text"])
    print(f"\n[완료] 저장 → {output_path}")
    print(f"파싱 성공: {success}/{len(results)}개")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 기존 공지사항 첨부파일 파싱
    # parse_all()

    # 2. 수동 업로드 파일 파싱
    parse_manual_files()