import os
import sys
import json
import zipfile
import subprocess
import shutil
import platform
import tempfile
import opendataloader_pdf
import pdfplumber
import pandas as pd
from docx import Document

# ── Windows 콘솔(cp949)에서도 한글/유니코드 출력 안전하게 ──────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACHMENT_DIR = os.path.join(BASE_DIR, "data", "attachments")
PARSED_DIR     = os.path.join(BASE_DIR, "data", "parsed")
RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
TMP_DIR        = os.path.join(BASE_DIR, "data", "tmp_convert")

os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── 상수 설정 ──────────────────────────────────────────────────────────────────

# 추출 텍스트가 이 값보다 적으면 스캔본/변환 실패 가능성 경고
EMPTY_TEXT_THRESHOLD = 50

# 스캔 PDF / 한글 문서가 많을 경우 hybrid=True, ocr_lang="ko,en" 으로 설정
# 단, hybrid 모드를 사용하려면 먼저 별도 터미널에서 서버를 실행해야 합니다:
#   pip install "opendataloader-pdf[hybrid]"
#   opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "ko,en"
USE_HYBRID      = False
HYBRID_BACKEND  = "docling-fast"

# opendataloader 결과에 Markdown 표가 이 수보다 많으면 pdfplumber 보강 생략
# 0으로 설정하면 항상 보강 (가장 안전)
TABLE_SUPPLEMENT_THRESHOLD = 0

# pdfplumber 표 추출 설정
PDFPLUMBER_TABLE_SETTINGS_LATTICE: dict = {
    "vertical_strategy":   "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance":       3,
    "join_tolerance":       3,
    "edge_min_length":      3,
    "min_words_vertical":   1,
    "min_words_horizontal": 1,
}
PDFPLUMBER_TABLE_SETTINGS_STREAM: dict = {
    "vertical_strategy":   "text",
    "horizontal_strategy": "text",
    "snap_tolerance":       3,
    "join_tolerance":       3,
}

# ── LibreOffice 탐색 ───────────────────────────────────────────────────────────

def find_libreoffice() -> str | None:
    """LibreOffice 실행 파일 탐색.
    1) PATH의 soffice  2) Windows 기본 설치 경로  3) None
    """
    found = shutil.which("soffice")
    if found:
        return found
    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None


LIBREOFFICE = find_libreoffice()

# 한컴 COM 자동화 가용 여부 (Windows + 한컴오피스 설치 시에만 True)
def _check_hancom_com() -> bool:
    """한컴오피스 COM 객체 등록 여부 확인."""
    if platform.system() != "Windows":
        return False
    try:
        import win32com.client
        win32com.client.Dispatch("HWPFrame.HwpObject")
        return True
    except Exception:
        return False

HANCOM_COM_AVAILABLE = _check_hancom_com()

SUPPORTED_EXTS = {".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".docx", ".txt", ".zip"}


# ── pdfplumber 표 추출 헬퍼 ────────────────────────────────────────────────────

def _table_to_markdown(table: list[list[str | None]], page_num: int, tbl_idx: int) -> str:
    """2D 리스트 형태의 표를 Markdown 표 문자열로 변환합니다.
    - None 셀 → 빈 문자열 처리
    - 빈 행 제거
    - 첫 행을 헤더로 사용
    """
    cleaned: list[list[str]] = []
    for row in table:
        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cell for cell in clean_row):
            cleaned.append(clean_row)

    if not cleaned:
        return ""

    max_cols   = max(len(row) for row in cleaned)
    normalized = [row + [""] * (max_cols - len(row)) for row in cleaned]
    header     = normalized[0]
    body       = normalized[1:]

    md_lines = [
        f"<!-- 표 p.{page_num}-{tbl_idx + 1} -->",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in body:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def _extract_tables_pdfplumber(file_path: str) -> list[str]:
    """pdfplumber로 PDF 내 모든 표를 Markdown 문자열 리스트로 추출합니다.

    추출 전략:
      1차) lattice: 셀 경계선이 명확한 표 (한국 공문서·보고서 형식에 적합)
      2차) stream:  lattice에서 발견되지 않으면 텍스트 간격 기반으로 재시도
    """
    tables_md: list[str] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                lattice_tables = page.extract_tables(
                    table_settings=PDFPLUMBER_TABLE_SETTINGS_LATTICE
                )
                page_tables = lattice_tables if lattice_tables else (
                    page.extract_tables(table_settings=PDFPLUMBER_TABLE_SETTINGS_STREAM)
                    or []
                )
                for tbl_idx, table in enumerate(page_tables):
                    if not table:
                        continue
                    md = _table_to_markdown(table, page_num, tbl_idx)
                    if md:
                        tables_md.append(md)
    except Exception as e:
        print(f"  [pdfplumber 표 추출 오류] {os.path.basename(file_path)}: {e}")
    return tables_md


def _count_md_tables(text: str) -> int:
    """Markdown 텍스트 내 표 개수를 구분선 행 기준으로 셉니다."""
    return text.count("| --- |") + text.count("|---|") + text.count("| ---")


def _supplement_tables(base_text: str, file_path: str) -> str:
    """opendataloader 출력에 pdfplumber 표를 보강합니다.

    - base_text의 Markdown 표 수가 TABLE_SUPPLEMENT_THRESHOLD 초과이면 생략
    - 그렇지 않으면 pdfplumber 표를 '보완 표' 섹션으로 추가
    """
    plumber_tables = _extract_tables_pdfplumber(file_path)
    if not plumber_tables:
        return base_text

    if _count_md_tables(base_text) > TABLE_SUPPLEMENT_THRESHOLD:
        return base_text

    supplement = (
        "\n\n---\n"
        "## [보완 표 — pdfplumber 추출]\n\n"
        + "\n\n".join(plumber_tables)
    )
    print(f"    → pdfplumber 표 {len(plumber_tables)}개 보강")
    return base_text + supplement


# ── PDF 파싱 (opendataloader-pdf + pdfplumber 보강) ────────────────────────────

def _odl_convert_batch(pdf_paths: list[str], output_dir: str) -> None:
    """opendataloader_pdf.convert() 래퍼. JVM 오버헤드 때문에 반드시 배치로 호출."""
    kwargs: dict = dict(
        input_path=pdf_paths,
        output_dir=output_dir,
        format="markdown",
    )
    if USE_HYBRID:
        kwargs["hybrid"] = HYBRID_BACKEND
    opendataloader_pdf.convert(**kwargs)


def parse_pdf_batch(pdf_map: dict[str, str]) -> dict[str, str]:
    """여러 PDF를 한 번의 JVM 호출로 파싱합니다 (권장 방식).

    동명 파일 충돌 방지를 위해 인덱스 prefix를 붙여 임시 디렉터리에 복사한 뒤
    변환하고, stem으로 원본 경로를 역추적합니다.
    각 파일에 대해 pdfplumber 표 보강을 추가로 수행합니다.

    Args:
        pdf_map: {원본_절대경로: 파일명} 딕셔너리

    Returns:
        {원본_절대경로: 추출된_텍스트} 딕셔너리
    """
    results: dict[str, str] = {}
    if not pdf_map:
        return results

    abs_paths = [os.path.abspath(p) for p in pdf_map.keys()]

    try:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir  = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            os.makedirs(out_dir)

            # 인덱스 prefix 붙여 복사 → 동명 파일 충돌 방지
            stem_to_orig: dict[str, str] = {}
            copied_inputs: list[str] = []
            for i, p in enumerate(abs_paths):
                base     = os.path.splitext(os.path.basename(p))[0]
                new_stem = f"{i:04d}_{base}"
                new_path = os.path.join(in_dir, new_stem + ".pdf")
                shutil.copyfile(p, new_path)
                stem_to_orig[new_stem] = p
                copied_inputs.append(new_path)

            _odl_convert_batch(copied_inputs, out_dir)

            for fname in os.listdir(out_dir):
                if not fname.endswith(".md"):
                    continue
                stem = os.path.splitext(fname)[0]
                orig = stem_to_orig.get(stem)
                if not orig:
                    continue
                try:
                    with open(os.path.join(out_dir, fname), encoding="utf-8") as f:
                        text = f.read().strip()
                    results[orig] = _supplement_tables(text, orig)
                except OSError as e:
                    print(f"  [PDF md 읽기 오류] {orig}: {e}")
                    results[orig] = ""

            # opendataloader가 출력을 생성하지 않은 파일 → pdfplumber 폴백
            for p in abs_paths:
                if p not in results:
                    print(f"  [PDF 경고] Markdown 출력 없음: {os.path.basename(p)}")
                    tables = _extract_tables_pdfplumber(p)
                    results[p] = "\n\n".join(tables) if tables else ""
                    if tables:
                        print(f"  [PDF 폴백] pdfplumber로 표 {len(tables)}개 추출")

    except Exception as e:
        print(f"  [PDF 배치 오류] {e}")
        # 배치 전체 실패 시 각 파일별 pdfplumber 폴백
        for p in abs_paths:
            if p not in results:
                try:
                    tables = _extract_tables_pdfplumber(p)
                    results[p] = "\n\n".join(tables) if tables else ""
                    if tables:
                        print(f"  [PDF 폴백] {os.path.basename(p)}: pdfplumber로 표 {len(tables)}개 추출")
                except Exception as e2:
                    print(f"  [PDF 폴백 오류] {p}: {e2}")
                    results.setdefault(p, "")

    return results


def parse_pdf(file_path: str) -> str:
    """단일 PDF 파싱. 내부적으로 parse_pdf_batch 호출 (JVM 1회 부팅 비용 발생)."""
    abs_path = os.path.abspath(file_path)
    return parse_pdf_batch({abs_path: os.path.basename(abs_path)}).get(abs_path, "")


# ── HWP / HWPX 파싱 ───────────────────────────────────────────────────────────

def _hancom_com_extract(file_path: str) -> str:
    """한컴오피스 COM 자동화로 HWP/HWPX 텍스트 추출 (Windows 전용).

    - 한컴오피스가 로컬에 설치되어 있어야 합니다.
    - 가장 정확한 방법으로, 표·각주·머리말 등 모든 요소를 처리합니다.
    - pip install pywin32  (최초 1회)

    동작 방식:
      HWPFrame.HwpObject COM 객체를 생성 → 파일 열기 → GetTextFile("TEXT") 로
      전체 텍스트 추출 → 오브젝트 종료. 프로세스 잔존을 막기 위해 반드시 Quit() 호출.
    """
    if not HANCOM_COM_AVAILABLE:
        return ""
    hwp = None
    try:
        import win32com.client
        hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
        # 보안 경고 팝업 억제
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        hwp.Open(os.path.abspath(file_path), "HWP", "forceopen:true")
        text = hwp.GetTextFile("TEXT", "")
        return text.strip() if text else ""
    except Exception as e:
        print(f"  [COM 오류] {os.path.basename(file_path)}: {e}")
        return ""
    finally:
        try:
            if hwp:
                hwp.Quit()
        except Exception:
            pass


def _parse_hwpx_native(file_path: str) -> str:
    """HWPX 파일을 ZIP+XML 구조로 직접 파싱합니다 (외부 도구 불필요).

    HWPX 포맷 구조:
      HWPX는 ZIP 아카이브이며, 본문은 Contents/section0.xml ~ sectionN.xml 에 위치.
      각 section XML의 텍스트 노드(hp:t 태그)를 순서대로 읽어 텍스트를 추출합니다.

    표 처리:
      hp:tbl(표) → hp:tr(행) → hp:tc(셀) 구조를 인식하여 Markdown 표로 변환합니다.

    한계:
      - 이미지, 수식, 그리기 개체 내부 텍스트는 추출되지 않습니다.
      - 복잡한 레이아웃(다단, 글상자)은 읽기 순서가 다소 어긋날 수 있습니다.
    """
    import xml.etree.ElementTree as ET

    # HWPX XML 네임스페이스
    NS = {
        "hp": "http://www.hancom.co.kr/hwpml/2012/paragraph",
        "hc": "http://www.hancom.co.kr/hwpml/2012/core",
        "hs": "http://www.hancom.co.kr/hwpml/2012/section",
    }

    def _elem_text(elem) -> str:
        """hp:t 태그에서 텍스트를 모아 반환합니다."""
        parts = []
        for t in elem.iter("{http://www.hancom.co.kr/hwpml/2012/paragraph}t"):
            if t.text:
                parts.append(t.text)
        return "".join(parts)

    def _parse_table(tbl_elem) -> str:
        """hp:tbl 요소를 Markdown 표로 변환합니다."""
        rows_md: list[str] = []
        tr_tag  = "{http://www.hancom.co.kr/hwpml/2012/paragraph}tr"
        tc_tag  = "{http://www.hancom.co.kr/hwpml/2012/paragraph}tc"
        for tr in tbl_elem.iter(tr_tag):
            cells = [_elem_text(tc).replace("\n", " ").strip() for tc in tr.iter(tc_tag)]
            if any(cells):
                rows_md.append("| " + " | ".join(cells) + " |")
        if not rows_md:
            return ""
        header    = rows_md[0]
        separator = "| " + " | ".join(["---"] * (header.count("|") - 1)) + " |"
        return "\n".join([header, separator] + rows_md[1:])

    result: list[str] = []
    try:
        with zipfile.ZipFile(file_path, "r") as z:
            section_files = sorted(
                name for name in z.namelist()
                if name.startswith("Contents/section") and name.endswith(".xml")
            )
            if not section_files:
                print(f"  [HWPX] section XML 없음: {os.path.basename(file_path)}")
                return ""

            p_tag   = "{http://www.hancom.co.kr/hwpml/2012/paragraph}p"
            tbl_tag = "{http://www.hancom.co.kr/hwpml/2012/paragraph}tbl"

            for sec_name in section_files:
                with z.open(sec_name) as f:
                    root = ET.parse(f).getroot()
                # 단락(p)과 표(tbl)를 문서 순서대로 처리
                for elem in root.iter():
                    tag = elem.tag
                    if tag == tbl_tag:
                        md_table = _parse_table(elem)
                        if md_table:
                            result.append(md_table)
                        # 표 내부 단락은 이미 처리했으므로 skip 처리를 위해 태그 제거
                        elem.tag = "__processed__"
                    elif tag == p_tag:
                        text = _elem_text(elem).strip()
                        if text:
                            result.append(text)

    except zipfile.BadZipFile:
        print(f"  [HWPX] ZIP 구조 아님 또는 손상된 파일: {os.path.basename(file_path)}")
        return ""
    except Exception as e:
        print(f"  [HWPX XML 파싱 오류] {os.path.basename(file_path)}: {e}")
        return ""

    return "\n".join(result)


def _libreoffice_convert(file_path: str, fmt: str, ext: str) -> str | None:
    """LibreOffice로 변환 후 결과 파일 경로 반환 (실패 시 None)."""
    if not LIBREOFFICE:
        print("  [오류] LibreOffice를 찾을 수 없습니다 (PATH 또는 표준 설치 경로 확인 필요).")
        return None
    try:
        subprocess.run(
            [LIBREOFFICE, "--headless", "--convert-to", fmt,
             "--outdir", TMP_DIR, file_path],
            timeout=60, capture_output=True, check=False
        )
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path  = os.path.join(TMP_DIR, base_name + "." + ext)
        return out_path if os.path.exists(out_path) else None
    except subprocess.TimeoutExpired:
        print(f"  [LibreOffice 타임아웃] {file_path}")
    except Exception as e:
        print(f"  [LibreOffice 변환 오류] {file_path}: {e}")
    return None


def _hwp5txt_extract(file_path: str) -> str:
    """pyhwp의 hwp5txt로 HWP에서 텍스트 추출 (LibreOffice fallback용)."""
    hwp5txt = shutil.which("hwp5txt")
    cmd = [hwp5txt, file_path] if hwp5txt else [
        sys.executable, "-m", "hwp5.hwp5txt", file_path
    ]
    try:
        res = subprocess.run(cmd, timeout=60, capture_output=True, check=False)
        if res.returncode != 0:
            return ""
        return res.stdout.decode("utf-8", errors="ignore").strip()
    except subprocess.TimeoutExpired:
        print(f"  [hwp5txt 타임아웃] {file_path}")
    except Exception as e:
        print(f"  [hwp5txt 오류] {file_path}: {e}")
    return ""


def parse_hwp(file_path: str) -> str:
    """HWP/HWPX 파싱: 4~5단계 fallback 전략.

    HWP 우선순위:
      1) 한컴오피스 COM 자동화  (Windows + 한컴오피스 설치 시, 가장 정확)
      2) LibreOffice → PDF → opendataloader  (표·레이아웃 보존)
      3) pyhwp hwp5txt  (LibreOffice가 거부하는 HWP에 강함)
      4) LibreOffice → TXT  (최후 수단)

    HWPX 우선순위:
      1) 한컴오피스 COM 자동화  (Windows + 한컴오피스 설치 시)
      2) HWPX 네이티브 XML 파싱  (외부 도구 불필요, 표 포함)
      3) LibreOffice → PDF → opendataloader
      4) LibreOffice → TXT
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ── 1단계: 한컴오피스 COM 자동화 (Windows 전용, 가장 정확) ──────────────
    if HANCOM_COM_AVAILABLE:
        text = _hancom_com_extract(file_path)
        if text:
            print("  [한컴 COM으로 추출 성공]")
            return text
        print("  [Fallback] COM 결과 비어 있음 - 다음 방법 시도")

    # ── 2단계 (HWPX 전용): 네이티브 XML 파싱 ─────────────────────────────────
    if ext == ".hwpx":
        text = _parse_hwpx_native(file_path)
        if text:
            print("  [HWPX 네이티브 XML로 추출 성공]")
            return text
        print("  [Fallback] HWPX XML 결과 비어 있음 - 다음 방법 시도")

    # ── 3단계: LibreOffice → PDF → opendataloader + pdfplumber 보강 ──────────
    pdf_path = _libreoffice_convert(file_path, "pdf", "pdf")
    if pdf_path:
        try:
            text = parse_pdf(pdf_path)
        except Exception as e:
            print(f"  [HWP→PDF 추출 실패] {e}")
            text = ""
        finally:
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        if text.strip():
            return text
        print("  [Fallback] HWP→PDF 결과 비어 있음 - 다음 방법 시도")

    # ── 4단계: pyhwp hwp5txt (.hwpx 미지원) ──────────────────────────────────
    if ext == ".hwp":
        text = _hwp5txt_extract(file_path)
        if text:
            print("  [pyhwp로 추출 성공]")
            return text

    # ── 5단계: LibreOffice → TXT (최후 수단) ─────────────────────────────────
    txt_path = _libreoffice_convert(file_path, "txt:Text", "txt")
    if not txt_path:
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"  [TXT 읽기 실패] {e}")
        return ""
    finally:
        try:
            os.remove(txt_path)
        except OSError:
            pass


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
    """ZIP 압축 해제 후 내부 파일 파싱.
    - zip-slip 공격 방지 (realpath 체크)
    - 내부 PDF는 배치 변환으로 JVM 호출 최소화
    """
    result: list[str] = []
    tmp_zip_dir = os.path.join(BASE_DIR, "data", "tmp_zip")
    try:
        os.makedirs(tmp_zip_dir, exist_ok=True)
        base_dir = os.path.realpath(tmp_zip_dir) + os.sep

        # zip-slip 방지: 추출 경로가 base 디렉터리를 벗어나는 엔트리 거부
        with zipfile.ZipFile(file_path, "r") as z:
            for member in z.infolist():
                target = os.path.realpath(os.path.join(tmp_zip_dir, member.filename))
                if not (target + os.sep).startswith(base_dir):
                    print(f"  [ZIP 위험 엔트리 거부] {member.filename}")
                    continue
                z.extract(member, tmp_zip_dir)

        # 내부 파일 수집 및 확장자별 분류
        pdf_map: dict[str, str] = {}
        non_pdf: list[tuple[str, str]] = []

        for root, _, files in os.walk(tmp_zip_dir):
            for fname in files:
                inner_path = os.path.join(root, fname)
                ext        = os.path.splitext(fname)[1].lower()
                if ext == ".pdf":
                    # 상대 경로를 값으로 저장 (출력 레이블용)
                    rel = os.path.relpath(inner_path, tmp_zip_dir)
                    pdf_map[os.path.abspath(inner_path)] = rel
                else:
                    non_pdf.append((inner_path, fname))

        # PDF 배치 변환 (pdfplumber 표 보강 포함)
        if pdf_map:
            pdf_results = parse_pdf_batch(pdf_map)
            for abs_path, rel_label in pdf_map.items():
                text = pdf_results.get(abs_path, "")
                if text:
                    result.append(f"[ZIP 내부: {rel_label}]\n{text}")

        # 나머지 파일 개별 파싱
        for inner_path, fname in non_pdf:
            ext  = os.path.splitext(fname)[1].lower()
            text = ""
            if ext in (".hwp", ".hwpx"):
                text = parse_hwp(inner_path)
            elif ext in (".xlsx", ".xls"):
                text = parse_xlsx(inner_path)
            elif ext == ".docx":
                text = parse_docx(inner_path)
            if text:
                rel = os.path.relpath(inner_path, tmp_zip_dir)
                result.append(f"[ZIP 내부: {rel}]\n{text}")

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
        return parse_hwp(file_path)
    elif ext in (".xlsx", ".xls"):
        return parse_xlsx(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".zip":
        return parse_zip(file_path)
    else:
        print(f"  [스킵] 지원하지 않는 형식: {file_path}")
        return ""


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def warn_if_empty(file_name: str, text: str) -> None:
    """추출 텍스트가 너무 적으면 경고 (스캔본 PDF 등 의심)."""
    if not text:
        print(f"  [!경고] 추출 실패 (0자): {file_name}")
    elif len(text) < EMPTY_TEXT_THRESHOLD:
        print(f"  [!경고] 추출 텍스트 매우 적음 ({len(text)}자) - 스캔본 가능성: {file_name}")


def _cleanup_tmp() -> None:
    """LibreOffice 임시 산출물 정리."""
    if os.path.exists(TMP_DIR):
        for f in os.listdir(TMP_DIR):
            try:
                os.remove(os.path.join(TMP_DIR, f))
            except OSError:
                pass


# ── 전체 공지사항 파싱 (증분) ─────────────────────────────────────────────────

def parse_all() -> None:
    """크롤링 첨부파일 파싱 (증분: 기존 결과가 있으면 재사용).
    PDF는 공지 전체에서 미처리 파일을 모아 한 번에 배치 변환합니다.
    """
    notices_path = os.path.join(RAW_DIR, "notices.json")
    output_path  = os.path.join(PARSED_DIR, "notices_parsed.json")

    with open(notices_path, encoding="utf-8") as f:
        notices = json.load(f)

    # 기존 파싱 결과 로드 (증분)
    existing: dict[str, dict[str, str]] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                prev = json.load(f)
            for n in prev:
                try:
                    wr = n["url"].split("wr_id=")[1].split("&")[0]
                except (KeyError, IndexError):
                    continue
                existing[wr] = {
                    a["name"]: a.get("parsed_text", "")
                    for a in n.get("attachments", []) if a.get("name")
                }
        except Exception:
            existing = {}

    print(f"[시작] 첨부파일 파싱 (공지 {len(notices)}개)")
    cached_count = 0
    parsed_count = 0

    # 미처리 PDF 전체 수집 → 배치 변환
    pdf_to_process: dict[str, str] = {}   # {절대경로: 파일명}
    for notice in notices:
        try:
            wr_id = notice["url"].split("wr_id=")[1].split("&")[0]
        except (KeyError, IndexError):
            continue
        attach_dir = os.path.join(ATTACHMENT_DIR, wr_id)
        for att in notice.get("attachments", []):
            file_name = att["name"]
            if os.path.splitext(file_name)[1].lower() != ".pdf":
                continue
            if existing.get(wr_id, {}).get(file_name):
                continue
            file_path = os.path.join(attach_dir, file_name)
            if os.path.exists(file_path):
                pdf_to_process[os.path.abspath(file_path)] = file_name

    pdf_results: dict[str, str] = {}
    if pdf_to_process:
        print(f"[PDF 배치] {len(pdf_to_process)}개 PDF 변환 중...")
        pdf_results = parse_pdf_batch(pdf_to_process)

    for notice in notices:
        try:
            wr_id = notice["url"].split("wr_id=")[1].split("&")[0]
        except (KeyError, IndexError):
            continue
        attach_dir  = os.path.join(ATTACHMENT_DIR, wr_id)
        attachments = notice.get("attachments", [])
        if not attachments:
            continue

        print(f"\n[{wr_id}] {notice['title'][:40]}")

        for att in attachments:
            file_name = att["name"]
            file_path = os.path.join(attach_dir, file_name)
            ext       = os.path.splitext(file_name)[1].lower()

            # 캐시 히트
            cached = existing.get(wr_id, {}).get(file_name, "")
            if cached:
                att["parsed_text"] = cached
                cached_count += 1
                print(f"  [캐시] {file_name} ({len(cached)}자)")
                continue

            if not os.path.exists(file_path):
                print(f"  [없음] {file_name}")
                continue

            if ext == ".pdf":
                text = pdf_results.get(os.path.abspath(file_path), "")
            else:
                print(f"  파싱 중: {file_name}")
                text = parse_file(file_path)

            att["parsed_text"] = text
            warn_if_empty(file_name, text)
            print(f"  완료: {len(text)}자 추출")
            parsed_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notices, f, ensure_ascii=False, indent=2)

    total_files = sum(len(n.get("attachments", [])) for n in notices)
    success     = sum(
        1 for n in notices for a in n.get("attachments", []) if a.get("parsed_text")
    )
    print(f"\n[완료] 파싱 결과 저장 → {output_path}")
    print(f"신규 파싱: {parsed_count}개 / 캐시 재사용: {cached_count}개 / 성공: {success}/{total_files}개")
    _cleanup_tmp()


# ── 수동 업로드 파일 파싱 (증분) ──────────────────────────────────────────────

def parse_manual_files() -> None:
    """manual_files 폴더의 파일들을 파싱합니다 (증분: 기존 결과 재사용).
    PDF는 한 번에 배치 변환합니다.
    """
    manual_dir  = os.path.join(BASE_DIR, "data", "manual_files")
    output_path = os.path.join(PARSED_DIR, "manual_parsed.json")

    # 기존 파싱 결과 로드 (증분)
    existing: dict[str, str] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                prev = json.load(f)
            existing = {
                r["file_name"]: r.get("parsed_text", "")
                for r in prev if r.get("file_name")
            }
        except Exception:
            existing = {}

    all_files = [
        fname for fname in sorted(os.listdir(manual_dir))
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS
    ]
    skipped = [
        fname for fname in os.listdir(manual_dir)
        if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTS
    ]
    for fname in skipped:
        print(f"  [스킵] {fname}")

    print(f"[시작] manual_files 파싱 ({len(all_files)}개 파일)")

    results: list[dict] = []
    cached_count = 0
    parsed_count = 0

    # 미처리 PDF 수집 → 배치 변환
    pdf_files: dict[str, str] = {}
    non_pdf: list[str] = []

    for fname in all_files:
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdf" and not existing.get(fname):
            pdf_files[os.path.abspath(os.path.join(manual_dir, fname))] = fname
        elif ext != ".pdf":
            non_pdf.append(fname)

    if pdf_files:
        print(f"[PDF 배치] {len(pdf_files)}개 PDF 변환 중...")
        pdf_results = parse_pdf_batch(pdf_files)
    else:
        pdf_results = {}

    for fname in all_files:
        fpath = os.path.join(manual_dir, fname)
        ext   = os.path.splitext(fname)[1].lower()

        # 캐시 히트
        cached = existing.get(fname)
        if cached:
            print(f"  [캐시] {fname} ({len(cached)}자)")
            results.append({"file_name": fname, "parsed_text": cached})
            cached_count += 1
            continue

        print(f"  파싱 중: {fname}")
        if ext == ".pdf":
            text = pdf_results.get(os.path.abspath(fpath), "")
        elif ext == ".txt":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        else:
            text = parse_file(fpath)

        warn_if_empty(fname, text)
        print(f"  완료: {len(text)}자 추출")
        results.append({"file_name": fname, "parsed_text": text})
        parsed_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in results if r["parsed_text"])
    print(f"\n[완료] 저장 → {output_path}")
    print(f"신규 파싱: {parsed_count}개 / 캐시 재사용: {cached_count}개 / 성공: {success}/{len(results)}개")
    _cleanup_tmp()


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not LIBREOFFICE:
        print("[경고] LibreOffice를 찾지 못했습니다. HWP/HWPX 파싱이 실패할 수 있습니다.")
        print("       - macOS/Linux: 'soffice' 명령이 PATH에 있는지 확인")
        print("       - Windows: C:\\Program Files\\LibreOffice\\program\\soffice.exe 설치 확인")
    else:
        print(f"[LibreOffice] {LIBREOFFICE}")

    if HANCOM_COM_AVAILABLE:
        print("[한컴 COM] 한컴오피스 COM 자동화 사용 가능 → HWP/HWPX 최고 품질로 파싱")
    else:
        print("[한컴 COM] 사용 불가 (Windows + 한컴오피스 설치 환경에서만 동작)")
        print("           HWPX는 네이티브 XML 파싱으로 대체됩니다.")

    # 1. 기존 공지사항 첨부파일 파싱
    # parse_all()

    # 2. 수동 업로드 파일 파싱
    parse_manual_files()