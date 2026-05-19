import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ChatSession, BookmarkedMessage } from '@/types'
import type { Message } from '@/types'

type SidebarTab = 'chat' | 'bookmarks'

interface SidebarStore {
  // ─── 상태 ─────────────────────────────────────────────────────────────
  isOpen: boolean
  activeTab: SidebarTab
  selectedSessionId: string | null
  chatSessions: ChatSession[]
  bookmarks: BookmarkedMessage[]
  isLoadingHistory: boolean

  // ─── 액션 ─────────────────────────────────────────────────────────────
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setActiveTab: (tab: SidebarTab) => void
  selectSession: (id: string) => void
  setChatSessions: (sessions: ChatSession[]) => void
  addSession: (session: ChatSession) => void
  updateSessionTitle: (id: string, title: string) => void
  removeSession: (id: string) => void
  setLoadingHistory: (loading: boolean) => void
  addBookmark: (message: Message, sessionId: string) => void
  removeBookmark: (messageId: string) => void
  isBookmarked: (messageId: string) => boolean
}

export const useSidebarStore = create<SidebarStore>()(
  persist(
    (set, get) => ({
      isOpen: true,
      activeTab: 'chat',
      selectedSessionId: null,
      chatSessions: [],
      bookmarks: [],
      isLoadingHistory: false,

      toggleSidebar: () => set((state) => ({ isOpen: !state.isOpen })),
      setSidebarOpen: (open) => set({ isOpen: open }),
      setActiveTab: (tab) => set({ activeTab: tab }),
      selectSession: (id) => set({ selectedSessionId: id, activeTab: 'chat' }),

      setChatSessions: (sessions) => set({ chatSessions: sessions }),

      addSession: (session) =>
        set((state) => ({ chatSessions: [session, ...state.chatSessions] })),

      updateSessionTitle: (id, title) =>
        set((state) => ({
          chatSessions: state.chatSessions.map((s) =>
            s.id === id ? { ...s, title } : s,
          ),
        })),

      removeSession: (id) =>
        set((state) => ({
          chatSessions: state.chatSessions.filter((s) => s.id !== id),
          selectedSessionId:
            state.selectedSessionId === id ? null : state.selectedSessionId,
        })),

      setLoadingHistory: (loading) => set({ isLoadingHistory: loading }),

      addBookmark: (message, sessionId) => {
        const already = get().isBookmarked(message.id)
        if (already) return
        const bookmark: BookmarkedMessage = {
          messageId: message.id,
          sessionId,
          content: message.content,
          sources: message.sources,
          bookmarkedAt: new Date().toISOString(),
        }
        set((state) => ({ bookmarks: [bookmark, ...state.bookmarks] }))
      },

      removeBookmark: (messageId) =>
        set((state) => ({
          bookmarks: state.bookmarks.filter((b) => b.messageId !== messageId),
        })),

      isBookmarked: (messageId) =>
        get().bookmarks.some((b) => b.messageId === messageId),
    }),
    {
      name: 'lucid-sidebar',
      // 즐겨찾기와 세션 목록만 로컬스토리지에 영속화
      partialize: (state) => ({
        bookmarks: state.bookmarks,
        chatSessions: state.chatSessions,
      }),
    },
  ),
)
