import os
import json
import zipfile
import subprocess
import shutil
import pdfplumber
import pandas as pd
from docx import Document

# 절대 경로로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACHMENT_DIR = os.path.join(BASE_DIR, "data", "attachments")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
TMP_DIR = os.path.join(BASE_DIR, "data", "tmp_convert")
LIBREOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
# 한글 to pdf 변환 라이브러리 사용 (반영필요)

os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def parse_pdf(file_path):
    """PDF에서 텍스트 + 표 추출"""
    result = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    for row in table:
                        row_text = " | ".join(
                            cell.strip() if cell else "" for cell in row
                        )
                        table_texts.append(row_text)

                page_content = text
                if table_texts:
                    page_content += "\n[표]\n" + "\n".join(table_texts)

                if page_content.strip():
                    result.append(page_content.strip())

    except Exception as e:
        print(f"  [PDF 오류] {file_path}: {e}")

    return "\n\n".join(result)


def hwp_to_txt(file_path):
    """LibreOffice로 HWP/HWPX → TXT 변환"""
    try:
        subprocess.run([
            LIBREOFFICE,
            "--headless",
            "--convert-to", "txt:Text",
            "--outdir", TMP_DIR,
            file_path
        ], timeout=30, capture_output=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_path = os.path.join(TMP_DIR, base_name + ".txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            os.remove(txt_path)
            return text.strip()
        else:
            print(f"  [HWP 변환 실패] txt 파일 없음: {base_name}")
            return ""

    except subprocess.TimeoutExpired:
        print(f"  [HWP 타임아웃] {file_path}")
        return ""
    except Exception as e:
        print(f"  [HWP 오류] {file_path}: {e}")
        return ""


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
    """ZIP 압축 해제 후 내부 파일 파싱"""
    result = []
    tmp_zip_dir = os.path.join(BASE_DIR, "data", "tmp_zip")
    try:
        os.makedirs(tmp_zip_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(tmp_zip_dir)

        for root, _, files in os.walk(tmp_zip_dir):
            for fname in files:
                inner_path = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                text = ""

                if ext == ".pdf":
                    text = parse_pdf(inner_path)
                elif ext in [".hwp", ".hwpx"]:
                    text = hwp_to_txt(inner_path)
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
        return hwp_to_txt(file_path)
    elif ext in [".xlsx", ".xls"]:
        return parse_xlsx(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".zip":
        return parse_zip(file_path)
    else:
        print(f"  [스킵] 지원하지 않는 형식: {file_path}")
        return ""


def parse_all():
    """전체 첨부파일 파싱 후 notices.json과 합쳐서 저장"""
    notices_path = os.path.join(RAW_DIR, "notices.json")
    with open(notices_path, encoding="utf-8") as f:
        notices = json.load(f)

    print(f"[시작] 첨부파일 파싱 (공지 {len(notices)}개)")

    for notice in notices:
        wr_id = notice["url"].split("wr_id=")[1].split("&")[0]
        attach_dir = os.path.join(ATTACHMENT_DIR, wr_id)
        attachments = notice.get("attachments", [])

        if not attachments:
            continue

        print(f"\n[{wr_id}] {notice['title'][:40]}")

        for att in attachments:
            file_name = att["name"]
            file_path = os.path.join(attach_dir, file_name)

            if not os.path.exists(file_path):
                print(f"  [없음] {file_name}")
                continue

            print(f"  파싱 중: {file_name}")
            text = parse_file(file_path)
            att["parsed_text"] = text
            print(f"  완료: {len(text)}자 추출")

    output_path = os.path.join(PARSED_DIR, "notices_parsed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notices, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 파싱 결과 저장 → {output_path}")

    total_files = 0
    parsed_files = 0
    for notice in notices:
        for att in notice.get("attachments", []):
            total_files += 1
            if att.get("parsed_text"):
                parsed_files += 1

    print(f"파싱 성공: {parsed_files}/{total_files}개")


def parse_manual_files():
    """manual_files 폴더의 파일들 파싱"""
    manual_dir = os.path.join(BASE_DIR, "data", "manual_files")
    output_path = os.path.join(PARSED_DIR, "manual_parsed.json")

    results = []
    print(f"[시작] manual_files 파싱")

    for fname in os.listdir(manual_dir):
        fpath = os.path.join(manual_dir, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext not in [".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".docx", ".txt", ".zip"]:
            print(f"  [스킵] {fname}")
            continue

        print(f"  파싱 중: {fname}")

        if ext == ".txt":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        else:
            text = parse_file(fpath)

        print(f"  완료: {len(text)}자 추출")
        results.append({
            "file_name": fname,
            "file_path": fpath,
            "parsed_text": text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 저장 → {output_path}")
    print(f"파싱 성공: {sum(1 for r in results if r['parsed_text'])}/{len(results)}개")


if __name__ == "__main__":
    # 1. 기존 공지사항 첨부파일 파싱
    # parse_all()

    # 2. 수동 업로드 파일 파싱
    parse_manual_files()