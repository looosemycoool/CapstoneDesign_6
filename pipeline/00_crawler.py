import requests
from bs4 import BeautifulSoup
import json
import os
import time

BASE_URL = "https://cse.knu.ac.kr/bbs/board.php"
PARAMS = {"bo_table": "sub5_1", "lang": "kor"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://cse.knu.ac.kr"
}

OUTPUT_DIR = "./data/raw"
ATTACHMENT_DIR = "./data/attachments"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ATTACHMENT_DIR, exist_ok=True)

# 세션 유지: 그누보드는 다운로드 시 세션 쿠키를 검증함
session = requests.Session()
session.headers.update(HEADERS)


def get_notice_list(page=1):
    """공지사항 목록 한 페이지 크롤링"""
    params = {**PARAMS, "page": page}
    try:
        res = session.get(BASE_URL, params=params, timeout=10)
        res.raise_for_status()
        return res.text
    except requests.exceptions.RequestException as e:
        print(f"[오류] 목록 페이지 {page} 요청 실패: {e}")
        return None


def parse_notice_list(html):
    """목록 페이지에서 게시글 링크와 제목 추출"""
    soup = BeautifulSoup(html, "html.parser")
    notices = []

    rows = soup.select("table tbody tr")
    for row in rows:
        title_div = row.select_one("div.bo_tit")
        if not title_div:
            continue

        target_link = None
        for a in title_div.select("a"):
            if "wr_id" in a.get("href", ""):
                target_link = a
                break

        if not target_link:
            continue

        title = target_link.get_text(strip=True)
        href = target_link.get("href", "")
        url = href if href.startswith("http") else "https://cse.knu.ac.kr" + href

        date_td = row.select_one("td.td_datetime")
        date = date_td.get_text(strip=True) if date_td else ""

        num_td = row.select_one("td.td_num2")
        num = num_td.get_text(strip=True) if num_td else ""

        notices.append({
            "num": num,
            "title": title,
            "url": url,
            "date": date
        })

    return notices


def get_notice_detail(url):
    """게시글 상세 내용 + 첨부파일 URL 수집"""
    try:
        res = session.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # 본문
        content_tag = soup.select_one("#bo_v_con")
        content = content_tag.get_text(separator="\n", strip=True) if content_tag else ""

        # 첨부파일
        attachments = []
        file_section = soup.select_one("section#bo_v_file")
        if file_section:
            for a_tag in file_section.select("a.view_file_download"):
                file_url = a_tag.get("href", "")
                strong_tag = a_tag.select_one("strong")
                file_name = strong_tag.get_text(strip=True) if strong_tag else a_tag.get_text(strip=True)
                if file_url:
                    attachments.append({
                        "name": file_name,
                        "url": file_url if file_url.startswith("http") else "https://cse.knu.ac.kr" + file_url
                    })

        return {
            "content": content,
            "attachments": attachments
        }

    except requests.exceptions.RequestException as e:
        print(f"[오류] 상세 페이지 요청 실패 ({url}): {e}")
        return None


def crawl(max_pages=3):
    """전체 크롤링 실행"""
    all_notices = []
    print(f"[시작] 경북대 컴퓨터학부 공지사항 크롤링 (최대 {max_pages}페이지)")

    for page in range(1, max_pages + 1):
        print(f"\n[페이지 {page}] 목록 수집 중...")
        html = get_notice_list(page)

        if html is None:
            print("[중단] 요청 실패")
            break

        notices = parse_notice_list(html)

        if not notices:
            print(f"[페이지 {page}] 파싱 실패")
            with open(f"debug_page{page}.html", "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  debug_page{page}.html 저장됨")
            break

        print(f"[페이지 {page}] {len(notices)}개 발견")

        for i, notice in enumerate(notices):
            print(f"  [{i+1}/{len(notices)}] {notice['title'][:40]}")
            detail = get_notice_detail(notice["url"])
            if detail:
                notice.update(detail)
            all_notices.append(notice)
            time.sleep(0.5)

    if all_notices:
        output_path = os.path.join(OUTPUT_DIR, "notices.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_notices, f, ensure_ascii=False, indent=2)
        print(f"\n[완료] {len(all_notices)}개 저장 → {output_path}")

    return all_notices


def download_attachments(notices):
    """수집된 공지사항의 첨부파일 전체 다운로드"""
    total = sum(len(n.get("attachments", [])) for n in notices)
    print(f"\n[다운로드 시작] 총 {total}개 첨부파일")

    downloaded = 0
    failed = 0
    skipped = 0

    SUPPORTED_EXT = {".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".docx", ".zip"}

    for notice in notices:
        attachments = notice.get("attachments", [])
        if not attachments:
            continue

        # 공지별 폴더 생성 (wr_id 기준)
        try:
            wr_id = notice["url"].split("wr_id=")[1].split("&")[0]
        except IndexError:
            continue

        save_dir = os.path.join(ATTACHMENT_DIR, wr_id)
        os.makedirs(save_dir, exist_ok=True)

        # 다운로드 전에 게시글 상세 페이지를 한 번 방문해서 세션 쿠키 확보
        notice_url = notice["url"]
        try:
            session.get(notice_url, timeout=10)
        except requests.exceptions.RequestException:
            pass  # 실패해도 다운로드는 시도

        for att in attachments:
            file_name = att["name"]
            file_url = att["url"]

            ext = os.path.splitext(file_name)[1].lower()
            if ext not in SUPPORTED_EXT:
                print(f"  [스킵] 지원하지 않는 형식: {file_name}")
                skipped += 1
                continue

            save_path = os.path.join(save_dir, file_name)

            if os.path.exists(save_path):
                print(f"  [스킵] 이미 존재: {file_name}")
                skipped += 1
                continue

            try:
                # Referer를 해당 게시글 상세 URL로 지정해야 다운로드 허용됨
                res = session.get(
                    file_url,
                    headers={"Referer": notice_url},
                    timeout=30
                )
                res.raise_for_status()

                # 응답이 실제 파일이 아닌 HTML(오류 페이지)인지 검증
                content_type = res.headers.get("Content-Type", "").lower()
                head = res.content[:512].lstrip().lower()
                is_html = (
                    "text/html" in content_type
                    or head.startswith(b"<!doctype html")
                    or head.startswith(b"<html")
                )
                if is_html:
                    print(f"  [실패] HTML 응답 (다운로드 거부됨): {file_name}")
                    failed += 1
                    continue

                with open(save_path, "wb") as f:
                    f.write(res.content)

                size_kb = len(res.content) / 1024
                print(f"  [완료] {file_name} ({size_kb:.1f}KB)")
                downloaded += 1
                time.sleep(0.3)

            except Exception as e:
                print(f"  [실패] {file_name}: {e}")
                failed += 1

    print(f"\n[다운로드 완료] 성공: {downloaded}개 / 스킵: {skipped}개 / 실패: {failed}개")
    print(f"저장 위치: {ATTACHMENT_DIR}")


if __name__ == "__main__":
    notices_path = "./data/raw/notices.json"

    # 1. 크롤링 (이미 있으면 기존 파일 로드)
    if os.path.exists(notices_path):
        print("[로드] 기존 크롤링 데이터 사용")
        with open(notices_path, encoding="utf-8") as f:
            notices = json.load(f)
    else:
        notices = crawl(max_pages=3)

    # 2. 첨부파일 다운로드
    download_attachments(notices)