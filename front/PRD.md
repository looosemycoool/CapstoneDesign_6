# 📋 PRD — 학교 공지사항 크롤링 기반 하이브리드 챗봇

> Product Requirements Document v1.0

---

## 1. 개요

### 배경

학교 포털 공지사항은 카테고리가 분산되어 있고 검색 기능이 불편해, 학생들이 원하는 정보를 찾는 데 시간이 많이 걸린다.

### 목적

크롤링으로 수집한 공지사항 데이터를 바탕으로 자연어로 질문하면 즉시 답변을 받을 수 있는 챗봇을 제공한다.

### 타겟 사용자

재학생 (공지 탐색 빈도가 높은 저학년 우선)

---

## 2. 핵심 기능 목록

### 2-1. 채팅 인터페이스

| 기능 | 설명 | 우선순위 |
|---|---|---|
| 메시지 전송 | 텍스트 입력 후 Enter 또는 전송 버튼으로 메시지 전송 | P0 |
| 스트리밍 응답 | 챗봇 응답을 토큰 단위로 실시간 렌더링 (타이핑 효과) | P0 |
| 마크다운 렌더링 | 응답 내 **굵기**, 목록, 링크 등을 렌더링 | P0 |
| 멀티턴 대화 | 이전 대화 맥락을 유지하여 후속 질문 처리 | P0 |
| 응답 중단 | 스트리밍 중 "중지" 버튼으로 응답 중단 | P1 |
| 메시지 복사 | 개별 메시지 복사 버튼 | P2 |

---

### 2-2. 공지사항 연동

| 기능 | 설명 | 우선순위 |
|---|---|---|
| 공지 출처 카드 | 답변 하단에 참조한 공지 제목 + 날짜 + 링크 카드 표시 | P0 |
| 출처 바로가기 | 카드 클릭 시 원문 공지 페이지로 이동 | P0 |
| 공지 최신화 여부 표시 | 크롤링 마지막 업데이트 시각 표시 | P1 |

---

### 2-3. 대화 히스토리

| 기능 | 설명 | 우선순위 |
|---|---|---|
| 대화 목록 (사이드바) | 날짜별 그룹핑된 이전 대화 목록 표시 | P0 |
| 대화 이어하기 | 히스토리에서 과거 대화 선택 시 해당 대화 복원 | P0 |
| 새 대화 시작 | 사이드바 상단 버튼으로 새 대화 생성 | P0 |
| 대화 삭제 | 개별 대화 삭제 | P1 |
| 대화 제목 자동 생성 | 첫 메시지 기반으로 대화 제목 자동 생성 | P1 |

---

### 2-4. 레이아웃 & UX

| 기능 | 설명 | 우선순위 |
|---|---|---|
| 사이드바 토글 | 사이드바 열기/닫기 | P0 |
| 첫 진입 환영 화면 | 대화 없을 때 예시 질문 카드 표시 | P0 |
| 자동 스크롤 | 새 메시지 도착 시 최하단 자동 스크롤 | P0 |
| 반응형 레이아웃 | 모바일에서 사이드바 오버레이 방식으로 전환 | P1 |
| 타이핑 인디케이터 | 응답 생성 중 점 애니메이션 표시 | P0 |
| 빈 상태 처리 | 검색 결과 없을 때 안내 메시지 표시 | P0 |
| 에러 처리 | API 실패 시 토스트 or 인라인 에러 메시지 | P0 |

---

## 3. 화면 구성

### 3-1. 전체 레이아웃

```
┌──────────────────────────────────────────────────────┐
│  [≡]  학교 공지 챗봇           [새 대화 +]            │  ← Header
├───────────────┬──────────────────────────────────────┤
│               │                                      │
│  사이드바      │         메시지 영역                   │
│               │                                      │
│  ─ 오늘 ──   │   [Assistant] 안녕하세요! ...         │
│  • 장학금 문의│                                      │
│  • 수강신청..│   [User] 2학기 수강신청 일정이 ...     │
│               │                                      │
│  ─ 어제 ──   │   [Assistant] 2학기 수강신청은 ...    │
│  • 졸업 요건  │   ┌──────────────────────────────┐   │
│               │   │ 📄 2025 수강신청 안내         │   │ ← 출처 카드
│               │   │ 학사지원팀 · 2025.06.10 →    │   │
│               │   └──────────────────────────────┘   │
│               │                                      │
├───────────────┴──────────────────────────────────────┤
│  [입력창 ............................................] [↑] │  ← Input
└──────────────────────────────────────────────────────┘
```

---

### 3-2. 첫 진입 환영 화면 (WelcomeScreen)

대화가 없을 때 중앙에 표시되는 화면.

```
        학교 공지 챗봇에 오신 걸 환영해요 👋
      궁금한 공지사항을 자연어로 질문해보세요.

  ┌─────────────────────┐  ┌─────────────────────┐
  │ 📅 수강신청 일정이    │  │ 💰 2학기 장학금      │
  │    언제야?           │  │    신청 방법 알려줘  │
  └─────────────────────┘  └─────────────────────┘
  ┌─────────────────────┐  ┌─────────────────────┐
  │ 📋 이번 학기 휴학    │  │ 🎓 졸업 요건이      │
  │    신청 기간 알려줘  │  │    어떻게 돼?        │
  └─────────────────────┘  └─────────────────────┘
```

---

### 3-3. 메시지 말풍선 (MessageBubble)

**User 메시지**
- 우측 정렬, 강조 배경색

**Assistant 메시지**
- 좌측 정렬, 아이콘 포함
- 마크다운 렌더링 지원
- 하단에 공지 출처 카드(AnnouncementCard) 첨부 가능
- 복사 버튼 (hover 시 표시)

---

## 4. 상태 관리 설계

### chatStore (Zustand)

```ts
interface ChatStore {
  // 현재 대화
  sessionId: string | null;
  messages: Message[];
  isStreaming: boolean;

  // 액션
  addMessage: (message: Message) => void;
  appendStreamChunk: (chunk: string) => void;
  setStreaming: (value: boolean) => void;
  resetChat: () => void;
}
```

### sidebarStore (Zustand)

```ts
interface SidebarStore {
  isOpen: boolean;
  selectedSessionId: string | null;

  toggleSidebar: () => void;
  selectSession: (id: string) => void;
}
```

---

## 5. API 인터페이스

### POST /api/chat

챗봇 메시지 전송 및 스트리밍 응답

```ts
// Request
interface ChatRequest {
  sessionId: string | null; // null이면 새 대화
  message: string;
}

// Response (SSE 스트리밍)
interface StreamChunk {
  type: "text" | "sources" | "done" | "error";
  content?: string;
  sources?: AnnouncementSource[];
}
```

### GET /api/history

대화 히스토리 목록 조회

```ts
interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}
```

### GET /api/history/:sessionId

특정 대화의 메시지 목록 조회

### DELETE /api/history/:sessionId

특정 대화 삭제

---

## 6. 타입 정의

```ts
// message.type.ts
type MessageRole = "user" | "assistant";

interface Message {
  id: string;
  role: MessageRole;
  content: string;
  sources?: AnnouncementSource[];
  createdAt: string;
}

// announcement.type.ts
interface AnnouncementSource {
  id: string;
  title: string;
  category: string;
  publishedAt: string;
  url: string;
}
```

---

## 7. 비기능 요구사항

| 항목 | 목표 |
|---|---|
| 초기 응답 시간 | 첫 토큰 수신까지 2초 이내 |
| 크롤링 주기 | 1일 1회 이상 |
| 접근성 | 키보드 전용 조작 가능 (Enter 전송, Tab 이동) |
| 브라우저 지원 | Chrome, Safari 최신 2버전 |

---

## 8. 개발 우선순위 (Phase)

### Phase 1 — MVP
- [ ] 기본 채팅 UI (입력 → 응답)
- [ ] 스트리밍 응답 렌더링
- [ ] 공지 출처 카드
- [ ] 에러 처리

### Phase 2 — 히스토리
- [ ] 대화 저장 및 목록
- [ ] 사이드바 구현
- [ ] 대화 이어하기

### Phase 3 — 완성도
- [ ] 반응형 모바일
- [ ] 예시 질문 환영 화면
- [ ] 응답 복사/중단
- [ ] 대화 제목 자동 생성
