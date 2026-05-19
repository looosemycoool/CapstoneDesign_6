import { useCallback } from 'react'
import { fetchHistory, fetchSessionDetail, deleteSession } from '@/api'
import { useChatStore, useSidebarStore } from '@/stores'

/**
 * 히스토리 관련 비즈니스 로직을 캡슐화하는 훅 (PRD 7-3)
 */
export function useChatHistory() {
  const { loadSession } = useChatStore()
  const {
    setChatSessions,
    removeSession,
    setLoadingHistory,
    setSidebarOpen,
    selectSession,
  } = useSidebarStore()

  /** 사이드바 대화 목록 불러오기 */
  const loadHistory = useCallback(async () => {
    setLoadingHistory(true)
    try {
      const { sessions } = await fetchHistory()
      setChatSessions(sessions)
    } catch (err) {
      console.error('히스토리 로드 실패:', err)
    } finally {
      setLoadingHistory(false)
    }
  }, [setChatSessions, setLoadingHistory])

  /** 특정 세션 클릭 → 메시지 복원 */
  const openSession = useCallback(
    async (sessionId: string) => {
      setLoadingHistory(true)
      try {
        const { messages, sessionId: id } = await fetchSessionDetail(sessionId)
        loadSession(messages, id)
        selectSession(id)
        // 모바일에서 세션 선택 시 사이드바 닫기
        if (window.innerWidth < 768) setSidebarOpen(false)
      } catch (err) {
        console.error('세션 불러오기 실패:', err)
      } finally {
        setLoadingHistory(false)
      }
    },
    [loadSession, selectSession, setSidebarOpen, setLoadingHistory],
  )

  /** 대화 삭제 */
  const removeChatSession = useCallback(
    async (sessionId: string) => {
      try {
        await deleteSession(sessionId)
        removeSession(sessionId)
      } catch (err) {
        console.error('세션 삭제 실패:', err)
        throw err
      }
    },
    [removeSession],
  )

  return { loadHistory, openSession, removeChatSession }
}
