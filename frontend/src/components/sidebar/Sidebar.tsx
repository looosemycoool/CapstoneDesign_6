import { SquarePen, Star } from 'lucide-react'
import { useChatStore, useSidebarStore } from '@/stores'
import { ChatHistoryList } from './ChatHistoryList'
import { BookmarkList } from './BookmarkList'
import { cn } from '@/lib/utils'

/**
 * 사이드바 (PRD 5. IA, F-HIST-01, F-HIST-06)
 * - [✏ 새글] 버튼
 * - [★ 내 보관함] 탭
 * - 채팅 목록 / 즐겨찾기 목록
 */
export function Sidebar() {
  const { activeTab, setActiveTab } = useSidebarStore()
  const { resetChat } = useChatStore()

  const handleNewChat = () => {
    resetChat()
    useSidebarStore.getState().selectSession('')
  }

  return (
    <div className="flex h-full flex-col bg-secondary/50 border-r">
      {/* 상단 액션 버튼 영역 */}
      <div className="flex items-center justify-between px-3 py-3 gap-2">
        <button
          onClick={handleNewChat}
          aria-label="새 대화 시작"
          className="flex flex-1 items-center gap-2 rounded-md px-3 py-2 text-sm font-medium hover:bg-accent transition-colors"
        >
          <SquarePen className="h-4 w-4" />
          새 대화
        </button>

        <button
          onClick={() => setActiveTab('bookmarks')}
          aria-label="내 보관함"
          aria-pressed={activeTab === 'bookmarks'}
          className={cn(
            'flex items-center gap-1 rounded-md px-3 py-2 text-sm font-medium transition-colors',
            activeTab === 'bookmarks'
              ? 'bg-knu text-white'
              : 'hover:bg-accent',
          )}
        >
          <Star className="h-4 w-4" />
          보관함
        </button>
      </div>

      {/* 탭 콘텐츠 */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {activeTab === 'chat' ? <ChatHistoryList /> : <BookmarkList />}
      </div>
    </div>
  )
}
