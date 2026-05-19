/**
 * 캠퍼스 줄임말 → 공식 명칭 매핑 테이블 (PRD F-UX-06)
 * 메시지 전송 직전 FE 파이프라인에서 치환 처리.
 * 새 줄임말 추가 시 이 파일만 수정하면 됨.
 */
export const SLANG_MAP: Record<string, string> = {
  컴학:   '컴퓨터학부',
  글솝:   '글로벌소프트웨어융합전공',
  전정:   '전자공학부',
  기공:   '기계공학부',
  인문대: '인문대학',
  사범대: '사범대학',
  공대:   '공과대학',
  자연대: '자연과학대학',
  경영대: '경영학부',
  행정:   '행정학부',
  수강신청: '수강신청',   // 정규식 혼동 방지용 identity 항목
}

/**
 * 입력 문자열에서 줄임말을 공식 명칭으로 치환하여 반환.
 * 단어 경계(word boundary) 기반으로 부분 매칭을 방지.
 */
export function replaceSlang(text: string): string {
  let result = text
  for (const [slang, official] of Object.entries(SLANG_MAP)) {
    // 한글은 \b가 동작하지 않으므로 공백/문장 경계 기반으로 처리
    const regex = new RegExp(`(?<![가-힣])${slang}(?![가-힣])`, 'g')
    result = result.replace(regex, official)
  }
  return result
}
