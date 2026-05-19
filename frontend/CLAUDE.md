# Lucid Frontend — CLAUDE.md

**스택**: TypeScript · React 18 · Vite · Tailwind CSS · Zustand · React Router · Axios · shadcn/ui · Lucide React

## 폴더 구조

`api/` API 호출 | `components/chat|layout|sidebar|ui/` UI | `constants/` 상수 | `hooks/` 커스텀 훅 | `lib/` 유틸 | `pages/` 페이지 | `stores/` Zustand | `types/` 타입

## 핵심 규칙

- Path alias: `@/` → `src/` (tsconfig + vite.config 동시 설정)
- 컴포넌트: Named Export, PascalCase.tsx / 페이지만 default export
- 스타일: `cn()` 유틸 사용, 인라인 style 금지 (동적 height 예외)
- API: REST → `axiosInstance`, SSE 스트리밍 → `fetch` + ReadableStream
- 상태: 전역 → Zustand, 로컬 UI → useState
- 보안: 마크다운 렌더링 시 `rehype-sanitize` 필수, 외부 링크 `rel="noopener noreferrer"`
- 접근성: 아이콘 버튼 `aria-label` 필수, 포커스 링 `focus-visible:ring-2`

## 개발 서버

```bash
cd frontend && npm install && npm run dev  # http://localhost:3000
```
