## 스토어
- `chatStore` : 현재 세션 메시지, 스트리밍 상태, AbortController
- `sidebarStore` : 사이드바 UI·채팅 목록·즐겨찾기 (localStorage 영속화)

## 규칙
- 액션은 순수함수, 비동기 로직은 `hooks/`에서 처리
- 상태는 반드시 setter를 통해서만 변경
- 파생 상태는 스토어에 저장하지 않고 컴포넌트에서 계산
- 스토어 간 접근은 `useXxxStore.getState()` snapshot 방식 사용
- 즐겨찾기 토글 시 `chatStore.toggleBookmark` + `sidebarStore.addBookmark` 동시 호출
