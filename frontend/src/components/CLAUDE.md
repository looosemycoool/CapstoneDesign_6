## 폴더 역할
`chat/` 메시지·입력·스트리밍 UI | `layout/` AppLayout·Header | `sidebar/` 히스토리·즐겨찾기 | `ui/` shadcn/ui 자동생성 (직접 편집 금지)

## 규칙
- Named Export, Props는 파일 상단 `interface Props {}` 선언
- 비즈니스 로직(API·스토어)은 `hooks/`로 분리, 컴포넌트는 렌더링만 담당
- 이벤트 핸들러: 내부 `handle{Action}`, Props로 받을 때 `on{Action}`
- `ui/` 추가는 `npx shadcn@latest add <component>` 명령만 사용

## chat/ 주의사항
- 마크다운 렌더링 시 `rehype-sanitize` 항상 포함
- `TypingIndicator`는 첫 SSE 토큰 수신 전 구간에만 표시 (MessageList 제어)
- 출처 아코디언은 `sources` 배열이 비어있으면 렌더링 생략
