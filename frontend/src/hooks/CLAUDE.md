## 훅 목록
- `useAutoScroll` : 스트리밍 중 자동 스크롤 + 사용자 스크롤 감지 (100px 임계값)
- `useChatHistory` : 히스토리 조회·세션 열기·세션 삭제
- `useSendMessage` : 메시지 전송, 줄임말 치환, SSE 수신, Abort 처리

## 규칙
- 훅 하나는 하나의 도메인 행동만 담당, JSX 반환 금지
- `useCallback` / `useMemo`는 의존성이 자주 변하지 않는 경우에 적용
- 에러는 `console.error` 기록 후 스토어 상태(isError 등)에 반영
- `AbortError`는 별도 분기하여 에러 UI 미표시 (사용자 의도적 중단)
