## 파일
- `message.type.ts` : `Message`, `AnnouncementSource`, `MessageRole`
- `chat.type.ts` : `ChatSession`, `BookmarkedMessage`, `StreamChunk`, API 요청/응답 타입
- `index.ts` : 전체 re-export

## 규칙
- 객체 형태는 `type` 대신 `interface` 우선 사용
- API 응답 타입은 PRD §9 백엔드 명세와 필드명 정확히 일치
- optional(`?`) 필드는 PRD에서 nullable로 명시된 경우에만 사용
- import 시 반드시 `import type` 구문 사용
- 타입 추가·변경 시 `index.ts` re-export 업데이트 필수
