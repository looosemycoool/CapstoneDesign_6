import os
from pathlib import Path

from dotenv import load_dotenv

import pandas as pd
import olefile
import zlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# =========================
# 0. 환경변수 로드
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")


# =========================
# 1. 경로 설정
# =========================
DATA_DIR = Path("./data")
PERSIST_DIR = "./chroma_db"

# =========================
# 1-1. HWP 추출 함수
# =========================
def extract_text_from_hwp(file_path: str) -> str:
    f = olefile.OleFileIO(file_path)
    dirs = f.listdir()

    if ["FileHeader"] not in dirs:
        raise ValueError("올바른 HWP 파일이 아닙니다.")

    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    text = []

    for d in dirs:
        if d[0] == "BodyText" and d[1].startswith("Section"):
            data = f.openstream("/".join(d)).read()

            if is_compressed:
                data = zlib.decompress(data, -15)

            i = 0
            while i < len(data):
                rec_header = int.from_bytes(data[i:i+4], "little")
                rec_type = rec_header & 0x3FF
                rec_len = (rec_header >> 20) & 0xFFF
                i += 4
                rec_data = data[i:i+rec_len]
                i += rec_len

                if rec_type == 67:
                    try:
                        text.append(rec_data.decode("utf-16"))
                    except:
                        pass

    return "\n".join(text)


# =========================
# 1-2. Excel 추출 함수
# =========================
def extract_text_from_excel(file_path: str) -> str:
    xls = pd.ExcelFile(file_path)
    all_text = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
        df = df.fillna("")
        sheet_text = f"[Sheet: {sheet_name}]\n"

        for _, row in df.iterrows():
            row_text = " | ".join([str(cell) for cell in row if str(cell).strip() != ""])
            if row_text.strip():
                sheet_text += row_text + "\n"

        all_text.append(sheet_text)

    return "\n".join(all_text)


# =========================
# 2. 파일 로드 함수
# =========================
def load_documents(data_dir: Path) -> list[Document]:
    docs = []

    for file_path in data_dir.glob("*"):
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = file_path.name
                docs.extend(loaded)

            elif suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = file_path.name
                docs.extend(loaded)

            elif suffix == ".hwp":
                text = extract_text_from_hwp(str(file_path))
                doc = Document(page_content=text, metadata={"source": file_path.name})
                docs.append(doc)

            elif suffix == ".xlsx":
                text = extract_text_from_excel(str(file_path))
                doc = Document(page_content=text, metadata={"source": file_path.name})
                docs.append(doc)

            else:
                print(f"[SKIP] 지원하지 않는 파일 형식: {file_path.name}")

        except Exception as e:
            print(f"[ERROR] {file_path.name} 로드 실패: {e}")

    return docs

# =========================
# 3. 문서 청크 분할
# =========================
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# =========================
# 4. 벡터DB 생성
# =========================
def build_vectorstore(split_docs: list[Document], embedding_model: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


# =========================
# 5. 질의응답 함수
# =========================
def ask_question(vectorstore, llm_model: str, question: str, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    print("DEBUG B: retriever 생성 완료")

    context = "\n\n".join([
        f"[출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    prompt = f"""
너는 경북대학교 컴퓨터학부 관련 문서를 바탕으로 답하는 도우미다.
반드시 제공된 문서 근거로 답해라.
문맥에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다."라고 답해라.

[문맥]
{context}

[질문]
{question}

[답변 규칙]
1. 질문에 대한 답만 간단명료하게 작성할 것
2. 문맥 기반으로만 답할 것
3. 숫자, 학점, 기준은 정확히 쓸 것
4. 불필요한 반복 설명 금지
5. 마지막 줄에 "참고 문서: 파일명" 형식으로만 적을 것
"""

    llm = ChatOpenAI(
        model=llm_model,
        temperature=0
    )

    response = llm.invoke(prompt)
    return response.content, retrieved_docs


# =========================
# 6. 메인 실행
# =========================
def main():
    embedding_model = "text-embedding-3-large"   # 나중에 large로 바꿔서 실험
    llm_model = "gpt-4o"                         # 나중에 o3-mini로 바꿔서 실험

    print("1) 문서 로드 중...")
    documents = load_documents(DATA_DIR)
    print(f"로드된 문서 수: {len(documents)}")

    print("2) 문서 분할 중...")
    split_docs = split_documents(documents)
    print(f"청크 수: {len(split_docs)}")

    print("3) 벡터DB 생성 중...")
    print("DEBUG A : 여기 도착")
    vectorstore = build_vectorstore(split_docs, embedding_model)
    print("벡터DB 생성 완료")

    # 테스트 질문 10개
    qa_pairs = [
        ("글로벌소프트웨어융합전공의 졸업에 필요한 총 학점과 최소 전공 학점은 각각 몇 학점인가?",
        "총 130학점 이상이며 최소 전공 학점은 51학점이다."),

        ("2023학년도 이후 입학생 기준 교양 학점 최소 기준과 초과 학점 처리 방식은 어떻게 되는가?",
        "교양 최소 이수 기준은 42학점이며 초과 학점은 교과구분 변경을 통해 전환할 수 있다."),

        ("글로벌소프트웨어융합전공의 전공 필수 과목에는 어떤 과목들이 포함되는가?",
        "전공 필수 과목은 글로벌소프트웨어융합전공 교육과정에 포함된 교과목으로 구성된다."),

        ("다중전공트랙에서 요구되는 영어성적, 해외학점, 창업 관련 조건을 모두 설명하라.",
        "공인 영어 성적과 함께 해외학점 이수 또는 창업 활동 등의 조건을 충족해야 한다."),

        ("다중전공트랙과 해외복수학위트랙의 영어 성적 기준과 필수 조건의 차이는 무엇인가?",
        "다중전공트랙은 기본 영어 성적과 추가 조건을 요구하며 해외복수학위트랙은 더 높은 영어 기준과 해외학점 이수를 요구한다."),

        ("현장실습과 해외어학연수로 취득한 학점은 졸업 전공 및 교양 학점에 어떻게 반영되는가?",
        "현장실습과 해외어학연수 학점은 인정되지만 최소 전공 및 교양 학점에는 포함되지 않는다."),

        ("부전공을 인정받기 위한 최소 학점 기준과 적용 조건은 무엇인가?",
        "부전공은 최소 이수 학점을 충족해야 하며 경우에 따라 전공 과목을 추가 이수해야 한다."),

        ("교양 42학점을 초과한 경우 교과구분 변경은 어떻게 신청하며 어떤 절차를 따르는가?",
        "교과구분 변경 신청서를 작성하여 제출하고 매뉴얼에 따라 승인 절차를 거친다."),

        ("스타트업 창업으로 졸업요건을 대체하기 위해 인정되는 조건에는 어떤 항목들이 포함되는가?",
        "창업경진대회 참여나 창업 실적 등이 일정 기준을 충족할 경우 인정된다."),

        ("졸업을 위해 요구되는 평점 기준과 조기졸업 조건은 어떻게 다른가?",
        "졸업 기준은 평점 1.7 이상이며 조기졸업은 평점 3.7 이상과 6학기 이상 이수 및 졸업요건 충족이 필요하다.")
    ]

    print("\n4) 질문 실행")

    for idx, (question, gt) in enumerate(qa_pairs, start=1):
        print("\n" + "=" * 60)
        print(f"[질문 {idx}] {question}")

        answer, retrieved_docs = ask_question(vectorstore, llm_model, question)

        print("\n[모델 답변]")
        print(answer)

        print("\n[정답]")
        print(gt)

        keywords = gt.split()

        matched = [k for k in keywords if k in answer]
        missing = [k for k in keywords if k not in answer]

        score = len(matched) >= len(keywords) * 0.5  # 절반 이상 맞으면 O

        print("\n[정확 여부]")
        print("O" if score else "X")

        print("\n[판단 근거]")
        if score:
            print(f"핵심 키워드 {len(matched)}/{len(keywords)} 포함됨 → 정답으로 판단")
            print("포함된 키워드:", matched[:5])
        else:
            print(f"핵심 키워드 부족 ({len(matched)}/{len(keywords)}) → 오답으로 판단")
            print("누락된 키워드:", missing[:5])

        # 참고 문서 1개만 출력
        print("\n[참고 문서]")
        if retrieved_docs:
            top_doc = retrieved_docs[0]
            print("문서명:", top_doc.metadata.get("source"))
            print("참고 내용:", top_doc.page_content[:150].replace("\n", " "))
        else:
            print("검색된 문서 없음")


if __name__ == "__main__":
    main()