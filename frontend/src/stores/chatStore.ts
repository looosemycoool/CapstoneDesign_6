import { create } from 'zustand'
import type { Message, AnnouncementSource } from '@/types'

interface ChatStore {
  // ─── 상태 ─────────────────────────────────────────────────────────────
  sessionId: string | null
  messages: Message[]
  isStreaming: boolean
  abortController: AbortController | null
  /** 에러 발생 시 재시도(F-UX-04)를 위해 직전 전송 메시지를 보존 */
  lastUserMessage: string | null

  // ─── 액션 ─────────────────────────────────────────────────────────────
  setSessionId: (id: string | null) => void
  addMessage: (message: Message) => void
  /** 스트리밍 중 어시스턴트 메시지 마지막 항목에 청크를 이어붙임 */
  appendStreamChunk: (chunk: string) => void
  /** 스트리밍 완료 시 sources 확정 및 isStreaming 해제 */
  finalizeStream: (sources?: AnnouncementSource[]) => void
  setStreaming: (value: boolean) => void
  setAbortController: (ctrl: AbortController | null) => void
  setLastUserMessage: (msg: string) => void
  /** 메시지 즐겨찾기 토글 */
  toggleBookmark: (messageId: string) => void
  /** 특정 메시지를 에러 상태로 변경 */
  setMessageError: (messageId: string) => void
  /** 세션 전면 초기화 (새 대화 시작) */
  resetChat: () => void
  /** 히스토리에서 세션 복원 */
  loadSession: (messages: Message[], sessionId: string) => void
}

const initialState = {
  sessionId: null,
  messages: [],
  isStreaming: false,
  abortController: null,
  lastUserMessage: null,
}

export const useChatStore = create<ChatStore>((set) => ({
  ...initialState,

  setSessionId: (id) => set({ sessionId: id }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  appendStreamChunk: (chunk) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === 'assistant') {
        messages[messages.length - 1] = { ...last, content: last.content + chunk }
      }
      return { messages }
    }),

  finalizeStream: (sources) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === 'assistant' && sources) {
        messages[messages.length - 1] = { ...last, sources }
      }
      return { messages, isStreaming: false, abortController: null }
    }),

  setStreaming: (value) => set({ isStreaming: value }),

  setAbortController: (ctrl) => set({ abortController: ctrl }),

  setLastUserMessage: (msg) => set({ lastUserMessage: msg }),

  toggleBookmark: (messageId) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId ? { ...m, isBookmarked: !m.isBookmarked } : m,
      ),
    })),

  setMessageError: (messageId) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId ? { ...m, isError: true } : m,
      ),
      isStreaming: false,
      abortController: null,
    })),

  resetChat: () => set(initialState),

  loadSession: (messages, sessionId) =>
    set({ messages, sessionId, isStreaming: false, abortController: null }),
}))
