import os
import sys
import json
import zipfile
import subprocess
import shutil
import platform
import tempfile
import opendataloader_pdf
import pandas as pd
from docx import Document

# Windows 콘솔(cp949)에서도 한글/유니코드 출력 안전하게
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 절대 경로로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACHMENT_DIR = os.path.join(BASE_DIR, "data", "attachments")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
TMP_DIR = os.path.join(BASE_DIR, "data", "tmp_convert")

os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# 추출 텍스트가 이 값보다 적으면 스캔본/변환 실패 가능성 경고
EMPTY_TEXT_THRESHOLD = 50


def find_libreoffice():
    """LibreOffice 실행 파일 탐색.
    1) PATH의 soffice 2) Windows 기본 설치 경로 3) None
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


def parse_pdf_batch(file_paths):
    """opendataloader_pdf로 여러 PDF를 한 번에 Markdown으로 변환.
    JVM 부팅이 호출당 ~수백ms라 배치가 필수. 반환: dict[절대경로]→markdown 텍스트.

    같은 basename을 가진 PDF가 다른 디렉터리에 있을 수 있으므로 (예: 서로 다른
    공지의 동명 첨부) 인덱스 prefix를 붙여 입력 디렉터리에 복사한 뒤 변환한다.
    """
    if not file_paths:
        return {}
    abs_paths = [os.path.abspath(p) for p in file_paths]
    results = {p: "" for p in abs_paths}
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # 인덱스 prefix를 붙여 복사 → md 파일 stem으로 원본 경로를 역추적
        stem_to_orig = {}
        copied_inputs = []
        for i, p in enumerate(abs_paths):
            base = os.path.splitext(os.path.basename(p))[0]
            new_stem = f"{i:04d}_{base}"
            new_path = os.path.join(in_dir, new_stem + ".pdf")
            try:
                shutil.copyfile(p, new_path)
            except OSError as e:
                print(f"  [PDF 복사 실패] {p}: {e}")
                continue
            stem_to_orig[new_stem] = p
            copied_inputs.append(new_path)

        if not copied_inputs:
            return results

        try:
            opendataloader_pdf.convert(
                input_path=copied_inputs, output_dir=out_dir, format="markdown"
            )
        except (RuntimeError, OSError, ValueError) as e:
            print(f"  [PDF 배치 변환 오류] {e}")
            return results

        for fname in os.listdir(out_dir):
            if not fname.endswith(".md"):
                continue
            stem = os.path.splitext(fname)[0]
            orig = stem_to_orig.get(stem)
            if not orig:
                continue
            try:
                with open(os.path.join(out_dir, fname), encoding="utf-8") as f:
                    results[orig] = f.read().strip()
            except OSError as e:
                print(f"  [PDF md 읽기 오류] {orig}: {e}")
    return results


def parse_pdf(file_path):
    """단일 PDF 파싱. 내부적으로 parse_pdf_batch 호출 (JVM 1회 부팅 비용 발생)."""
    abs_path = os.path.abspath(file_path)
    return parse_pdf_batch([abs_path]).get(abs_path, "")


def _libreoffice_convert(file_path, fmt, ext):
    """LibreOffice로 변환 후 결과 파일 경로 반환 (실패 시 None)"""
    if not LIBREOFFICE:
        print("  [오류] LibreOffice를 찾을 수 없습니다 (PATH 또는 표준 설치 경로 확인 필요).")
        return None
    try:
        subprocess.run([
            LIBREOFFICE,
            "--headless",
            "--convert-to", fmt,
            "--outdir", TMP_DIR,
            file_path
        ], timeout=60, capture_output=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(TMP_DIR, base_name + "." + ext)
        return out_path if os.path.exists(out_path) else None
    except subprocess.TimeoutExpired:
        print(f"  [HWP 타임아웃] {file_path}")
        return None
    except Exception as e:
        print(f"  [HWP 변환 오류] {file_path}: {e}")
        return None


def _hwp5txt_extract(file_path):
    """pyhwp의 hwp5txt로 HWP에서 텍스트 추출 (LibreOffice가 못 읽는 파일용 fallback)"""
    hwp5txt = shutil.which("hwp5txt")
    cmd = [hwp5txt, file_path] if hwp5txt else [
        sys.executable, "-m", "hwp5.hwp5txt", file_path
    ]
    try:
        res = subprocess.run(cmd, timeout=60, capture_output=True)
        if res.returncode != 0:
            return ""
        return res.stdout.decode("utf-8", errors="ignore").strip()
    except subprocess.TimeoutExpired:
        print(f"  [hwp5txt 타임아웃] {file_path}")
        return ""
    except Exception as e:
        print(f"  [hwp5txt 오류] {file_path}: {e}")
        return ""


def parse_hwp(file_path):
    """HWP/HWPX 파싱: LibreOffice→PDF→PyMuPDF 우선, 실패 시 pyhwp로 fallback.
    .hwpx는 pyhwp가 지원 안 하므로 LibreOffice 경로만 사용."""
    ext = os.path.splitext(file_path)[1].lower()

    # 1차: LibreOffice → PDF → PyMuPDF (표/레이아웃 보존에 유리)
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
        print(f"  [Fallback] HWP→PDF 결과 비어 있음 - 다음 방법 시도")

    # 2차: pyhwp (LibreOffice가 거부하는 HWP 파일에 강함). .hwpx는 미지원.
    if ext == ".hwp":
        text = _hwp5txt_extract(file_path)
        if text:
            print(f"  [pyhwp로 추출 성공]")
            return text

    # 3차: LibreOffice → TXT
    txt_path = _libreoffice_convert(file_path, "txt:Text", "txt")
    if not txt_path:
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text.strip()
    except Exception as e:
        print(f"  [TXT 읽기 실패] {e}")
        return ""
    finally:
        try:
            os.remove(txt_path)
        except OSError:
            pass


def parse_xlsx(file_path):
    """엑셀 파일에서 텍스트 추출"""
    result = []
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


def parse_docx(file_path):
    """DOCX에서 텍스트 추출"""
    result = []
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


def parse_zip(file_path):
    """ZIP 압축 해제 후 내부 파일 파싱. 내부 PDF는 한 번에 배치 처리."""
    result = []
    tmp_zip_dir = os.path.join(BASE_DIR, "data", "tmp_zip")
    try:
        os.makedirs(tmp_zip_dir, exist_ok=True)
        # zip-slip 방지: 추출 경로가 base 디렉터리를 벗어나는 엔트리는 거부
        base_dir = os.path.realpath(tmp_zip_dir) + os.sep
        with zipfile.ZipFile(file_path, "r") as z:
            for member in z.infolist():
                target = os.path.realpath(os.path.join(tmp_zip_dir, member.filename))
                if not (target + os.sep).startswith(base_dir):
                    print(f"  [ZIP 위험 엔트리 거부] {member.filename}")
                    continue
                z.extract(member, tmp_zip_dir)

        # 내부 파일 수집 (확장자별 분류)
        inner_files = []
        for root, _, files in os.walk(tmp_zip_dir):
            for fname in files:
                inner_files.append((fname, os.path.join(root, fname)))
        pdf_paths = [p for f, p in inner_files if os.path.splitext(f)[1].lower() == ".pdf"]
        pdf_results = parse_pdf_batch(pdf_paths) if pdf_paths else {}

        for fname, inner_path in inner_files:
            ext = os.path.splitext(fname)[1].lower()
            text = ""
            if ext == ".pdf":
                text = pdf_results.get(os.path.abspath(inner_path), "")
            elif ext in [".hwp", ".hwpx"]:
                text = parse_hwp(inner_path)
            elif ext in [".xlsx", ".xls"]:
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


def parse_file(file_path):
    """파일 확장자에 따라 적절한 파서 호출"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in [".hwp", ".hwpx"]:
        return parse_hwp(file_path)
    elif ext in [".xlsx", ".xls"]:
        return parse_xlsx(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".zip":
        return parse_zip(file_path)
    else:
        print(f"  [스킵] 지원하지 않는 형식: {file_path}")
        return ""


def warn_if_empty(file_name, text):
    """추출 텍스트가 너무 적으면 경고 (스캔본 PDF 등 의심)"""
    if not text:
        print(f"  [!경고] 추출 실패 (0자): {file_name}")
    elif len(text) < EMPTY_TEXT_THRESHOLD:
        print(f"  [!경고] 추출 텍스트 매우 적음 ({len(text)}자) - 스캔본 가능성: {file_name}")


def parse_all():
    """크롤링 첨부파일 파싱 (증분: 기존 결과가 있으면 재사용)"""
    notices_path = os.path.join(RAW_DIR, "notices.json")
    output_path = os.path.join(PARSED_DIR, "notices_parsed.json")

    with open(notices_path, encoding="utf-8") as f:
        notices = json.load(f)

    # 기존 파싱 결과 로드 (증분 파싱)
    existing = {}
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

    # PDF 배치 변환을 위해 처리 대상 PDF 경로 먼저 수집
    pdf_to_process = []
    for notice in notices:
        try:
            wr_id = notice["url"].split("wr_id=")[1].split("&")[0]
        except (KeyError, IndexError):
            continue
        attach_dir = os.path.join(ATTACHMENT_DIR, wr_id)
        for att in notice.get("attachments", []):
            file_name = att["name"]
            ext = os.path.splitext(file_name)[1].lower()
            if ext != ".pdf":
                continue
            cached = existing.get(wr_id, {}).get(file_name, "")
            if cached:
                continue
            file_path = os.path.join(attach_dir, file_name)
            if os.path.exists(file_path):
                pdf_to_process.append(file_path)

    pdf_results = {}
    if pdf_to_process:
        print(f"[PDF 배치] {len(pdf_to_process)}개 PDF를 opendataloader로 변환 중...")
        pdf_results = parse_pdf_batch(pdf_to_process)

    for notice in notices:
        try:
            wr_id = notice["url"].split("wr_id=")[1].split("&")[0]
        except (KeyError, IndexError):
            continue
        attach_dir = os.path.join(ATTACHMENT_DIR, wr_id)
        attachments = notice.get("attachments", [])

        if not attachments:
            continue

        print(f"\n[{wr_id}] {notice['title'][:40]}")

        for att in attachments:
            file_name = att["name"]
            file_path = os.path.join(attach_dir, file_name)
            ext = os.path.splitext(file_name)[1].lower()

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
    success = sum(
        1 for n in notices for a in n.get("attachments", []) if a.get("parsed_text")
    )

    print(f"\n[완료] 파싱 결과 저장 → {output_path}")
    print(f"신규 파싱: {parsed_count}개 / 캐시 재사용: {cached_count}개 / 성공: {success}/{total_files}개")
    _cleanup_tmp()


def parse_manual_files():
    """manual_files 폴더의 파일들 파싱 (증분)"""
    manual_dir = os.path.join(BASE_DIR, "data", "manual_files")
    output_path = os.path.join(PARSED_DIR, "manual_parsed.json")

    existing = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                prev = json.load(f)
            existing = {r["file_name"]: r.get("parsed_text", "") for r in prev if r.get("file_name")}
        except Exception:
            existing = {}

    results = []
    cached_count = 0
    parsed_count = 0
    print(f"[시작] manual_files 파싱")

    # PDF 배치 변환 대상 수집
    pdf_to_process = []
    for fname in sorted(os.listdir(manual_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdf" and not (fname in existing and existing[fname]):
            pdf_to_process.append(os.path.join(manual_dir, fname))

    pdf_results = {}
    if pdf_to_process:
        print(f"[PDF 배치] {len(pdf_to_process)}개 PDF를 opendataloader로 변환 중...")
        pdf_results = parse_pdf_batch(pdf_to_process)

    for fname in sorted(os.listdir(manual_dir)):
        fpath = os.path.join(manual_dir, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext not in [".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".docx", ".txt", ".zip"]:
            print(f"  [스킵] {fname}")
            continue

        if fname in existing and existing[fname]:
            print(f"  [캐시] {fname} ({len(existing[fname])}자)")
            results.append({"file_name": fname, "parsed_text": existing[fname]})
            cached_count += 1
            continue

        if ext == ".pdf":
            text = pdf_results.get(os.path.abspath(fpath), "")
        elif ext == ".txt":
            print(f"  파싱 중: {fname}")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        else:
            print(f"  파싱 중: {fname}")
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


def _cleanup_tmp():
    """LibreOffice 임시 산출물 정리"""
    if os.path.exists(TMP_DIR):
        for f in os.listdir(TMP_DIR):
            try:
                os.remove(os.path.join(TMP_DIR, f))
            except OSError:
                pass


if __name__ == "__main__":
    if not LIBREOFFICE:
        print("[경고] LibreOffice를 찾지 못했습니다. HWP/HWPX 파싱이 실패할 수 있습니다.")
        print("       - macOS/Linux: 'soffice' 명령이 PATH에 있는지 확인")
        print("       - Windows: C:\\Program Files\\LibreOffice\\program\\soffice.exe 설치 확인")
    else:
        print(f"[LibreOffice] {LIBREOFFICE}")

    # 1. 기존 공지사항 첨부파일 파싱
    # parse_all()

    # 2. 수동 업로드 파일 파싱
    parse_manual_files()
