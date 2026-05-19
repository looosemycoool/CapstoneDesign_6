## 파일

- `chat.constant.ts` : 추천 질문 카드, 타임아웃, 스크롤 임계값 등 숫자·배열 상수
- `slang.constant.ts` : 캠퍼스 줄임말 → 공식 명칭 매핑 + `replaceSlang()` 함수

## 규칙

- 상수명 `UPPER_SNAKE_CASE`, 단위 포함 (예: `_MS`, `_PX`)
- `as const` 어서션으로 리터럴 타입 고정
- 줄임말 추가는 `SLANG_MAP` 객체에만 항목 추가 (`replaceSlang` 수정 불필요)
- 매직 넘버·문자열을 컴포넌트·훅에 직접 작성 금지, 반드시 이 폴더에서 관리
