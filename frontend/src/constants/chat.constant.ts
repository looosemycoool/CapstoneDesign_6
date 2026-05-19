/** 웰컴 화면 추천 질문 카드 목록 (PRD F-INIT-03) */
export const EXAMPLE_QUESTIONS = [
  { emoji: '📅', text: '수강신청 일정 언제야?' },
  { emoji: '🎓', text: '졸업 요건 알려줘.' },
  { emoji: '🏫', text: '장학금 신청 방법 알려줘.' },
  { emoji: '🏠', text: '기숙사 신청은 어떻게 해?' },
] as const

/** 스트리밍 타임아웃 (ms) — PRD: 30s */
export const STREAM_TIMEOUT_MS = 30_000

/** 자동 스크롤 일시정지 임계값 (px) — PRD F-UX-02 */
export const AUTO_SCROLL_THRESHOLD_PX = 100

/** 즐겨찾기 복사 완료 피드백 표시 시간 (ms) — PRD F-CHAT-04 */
export const COPY_FEEDBACK_MS = 1_500

/** 대화방 제목 최대 길이 (자) — PRD F-HIST-02 */
export const SESSION_TITLE_MAX_LENGTH = 20

/** 출처 카드 최대 표시 개수 — PRD F-SRC-02 */
export const MAX_SOURCE_CARDS = 5
