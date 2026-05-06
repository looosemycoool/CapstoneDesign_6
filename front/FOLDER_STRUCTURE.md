# 📁 프로젝트 폴더 구조

> 학교 공지사항 크롤링 기반 하이브리드 챗봇 — React + TypeScript + Tailwind

---

## 전체 구조

```
src/
├── pages/                        # 라우트 단위 페이지 (UI 조립만 담당, API 직접 호출 ❌)
│   ├── chat/
│   │   ├── ChatPage.tsx          # 메인 챗봇 화면
│   │   └── index.ts
│   └── not-found/
│       ├── NotFoundPage.tsx
│       └── index.ts
│
├── components/                   # 재사용 UI 컴포넌트 (비즈니스 로직 ❌)
│   ├── layout/
│   │   ├── Sidebar.tsx           # 대화 히스토리 목록
│   │   ├── Header.tsx            # 헤더 (로고, 새 대화 버튼 등)
│   │   └── AppLayout.tsx         # 전체 레이아웃 래퍼
│   │
│   ├── chat/
│   │   ├── MessageBubble.tsx     # 개별 메시지 말풍선 (user / assistant 구분)
│   │   ├── MessageList.tsx       # 메시지 스크롤 영역
│   │   ├── ChatInput.tsx         # 입력창 + 전송 버튼
│   │   ├── TypingIndicator.tsx   # 스트리밍 중 로딩 애니메이션
│   │   ├── WelcomeScreen.tsx     # 첫 진입 시 안내 화면 (예시 질문 포함)
│   │   └── AnnouncementCard.tsx  # 공지 출처 카드 (카드 형태 답변 렌더링)
│   │
│   └── common/
│       ├── Button.tsx
│       ├── IconButton.tsx
│       ├── Spinner.tsx
│       └── ErrorBoundary.tsx
│
├── hooks/                        # 커스텀 훅 (상태 + 로직 + API 호출)
│   ├── useChat.ts                # 메시지 송수신, 스트리밍 처리
│   ├── useChatHistory.ts         # 대화 목록 불러오기 / 삭제
│   ├── useAnnouncements.ts       # 공지사항 목록 조회
│   └── useAutoScroll.ts          # 메시지 추가 시 자동 스크롤
│
├── api/                          # 서버 통신만 담당 (네트워크 로직만 포함)
│   ├── chat.api.ts               # 챗봇 메시지 전송 / 스트리밍 API
│   ├── history.api.ts            # 대화 히스토리 CRUD
│   └── announcement.api.ts       # 크롤링된 공지 조회 API
│
├── store/                        # Zustand 전역 상태
│   ├── chatStore.ts              # 현재 대화 메시지 목록, 로딩 상태
│   └── sidebarStore.ts          # 사이드바 열림/닫힘, 선택된 대화 ID
│
├── types/                        # 타입 정의
│   ├── message.type.ts           # Message, MessageRole, StreamChunk 등
│   ├── chat.type.ts              # ChatSession, ChatHistory 등
│   └── announcement.type.ts      # Announcement, AnnouncementSource 등
│
├── constants/                    # 상수 값 모음
│   ├── api.constant.ts           # API base URL, endpoint 경로
│   └── chat.constant.ts          # 역할 구분 상수, 예시 질문 목록 등
│
├── utils/                        # 순수 함수
│   ├── formatDate.ts             # 날짜 포맷 (공지 날짜 표시용)
│   ├── parseMarkdown.ts          # 챗봇 응답 마크다운 파싱
│   └── groupByDate.ts            # 히스토리 날짜별 그룹핑
│
├── styles/
│   └── global.css                # Tailwind 디렉티브 + 글로벌 스타일
│
├── lib/
│   └── axios.ts                  # axios 인스턴스 (baseURL, interceptor 설정)
│
└── main.tsx
```

---

## 역할 분리 규칙

| 레이어 | 역할 | 제한 |
|---|---|---|
| `pages/` | 화면 단위, UI 조립 | API 직접 호출 ❌ |
| `components/` | 재사용 UI | 비즈니스 로직 ❌ |
| `hooks/` | 상태 + 로직 + API 호출 | - |
| `api/` | 서버 통신 | 네트워크 로직만 |
| `store/` | 전역 상태 (Zustand) | 비즈니스 로직 최소화 |
| `constants/` | 매직 넘버/문자열 제거 | - |
| `utils/` | 순수 함수 | 사이드 이펙트 ❌ |

---

## 네이밍 규칙

| 분류 | 규칙 | 예시 |
|---|---|---|
| 폴더 | kebab-case | `chat/`, `not-found/` |
| 컴포넌트 | PascalCase.tsx | `MessageBubble.tsx` |
| 커스텀 훅 | useXxx.ts | `useChat.ts` |
| API 파일 | xxx.api.ts | `chat.api.ts` |
| 타입 파일 | xxx.type.ts | `message.type.ts` |
| 상수 파일 | xxx.constant.ts | `chat.constant.ts` |
| Zustand 스토어 | xxxStore.ts | `chatStore.ts` |
