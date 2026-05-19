## 파일
- `axiosInstance.ts` : 공통 axios 설정 (baseURL `/api`, timeout 10s)
- `chat.api.ts` : `POST /api/chat` — fetch + SSE ReadableStream (axios 금지)
- `history.api.ts` : `GET|DELETE /api/history` — axios
- `meta.api.ts` : `GET /api/meta` — axios

## 규칙
- SSE는 반드시 `fetch`로 처리, `AbortController.signal` 전달 필수
- 청크 타입 처리 순서: `text` → `session` → `sources` → `done` / `error`
- API 함수는 에러를 throw만 하고, UI 피드백은 호출하는 `hooks/`에서 담당
- 새 엔드포인트 추가 시 `types/`에 타입 먼저 정의 후 `index.ts` re-export
