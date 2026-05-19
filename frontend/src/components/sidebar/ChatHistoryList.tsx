import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import { useSidebarStore } from '@/stores'
import { useChatHistory } from '@/hooks'
import { getDateGroup } from '@/lib/utils'
import { cn } from '@/lib/utils'

/**
 * 날짜별 그룹으로 묶인 채팅 히스토리 목록 (PRD F-HIST-02, F-HIST-03, F-HIST-04)
 */
export function ChatHistoryList() {
  const { chatSessions, selectedSessionId, isLoadingHistory, setActiveTab } =
    useSidebarStore()
  const { openSession, removeChatSession } = useChatHistory()
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  if (isLoadingHistory) {
    return (
      <div className="px-3 py-4 text-xs text-muted-foreground">불러오는 중...</div>
    )
  }

  if (chatSessions.length === 0) {
    return (
      <div className="px-3 py-4 text-xs text-muted-foreground">대화 기록이 없습니다.</div>
    )
  }

  // 날짜 그룹화
  const groups: Record<string, typeof chatSessions> = {}
  const ORDER = ['오늘', '어제', '이전 7일', '더 오래된 대화']

  for (const session of chatSessions) {
    const label = getDateGroup(session.updatedAt)
    if (!groups[label]) groups[label] = []
    groups[label].push(session)
  }

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    try {
      await removeChatSession(id)
    } catch {
      // 삭제 실패 시 토스트 등 추가 가능
    } finally {
      setPendingDelete(null)
    }
  }

  return (
    <div className="px-2 py-2 space-y-4">
      {ORDER.filter((g) => groups[g]).map((group) => (
        <div key={group}>
          <p className="px-2 mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            {group}
          </p>
          <ul>
            {groups[group].map((session) => (
              <li key={session.id} className="group relative">
                <button
                  onClick={() => {
                    openSession(session.id)
                    setActiveTab('chat')
                  }}
                  className={cn(
                    'w-full truncate rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent',
                    selectedSessionId === session.id && 'bg-accent font-medium',
                  )}
                >
                  {session.title || '새 대화'}
                </button>

                {/* 삭제 버튼 — hover 시 노출 */}
                {pendingDelete === session.id ? (
                  <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-1">
                    <button
                      onClick={(e) => handleDelete(e, session.id)}
                      className="rounded px-2 py-0.5 text-xs bg-destructive text-destructive-foreground"
                    >
                      삭제
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); setPendingDelete(null) }}
                      className="rounded px-2 py-0.5 text-xs border"
                    >
                      취소
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={(e) => { e.stopPropagation(); setPendingDelete(session.id) }}
                    aria-label="대화 삭제"
                    className="absolute right-2 top-1/2 -translate-y-1/2 hidden rounded p-1 hover:text-destructive group-hover:flex"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
