import { Star } from 'lucide-react'
import { useSidebarStore } from '@/stores'
import { formatDateTime } from '@/lib/utils'

/**
 * 내 보관함 탭 — 즐겨찾기된 답변 목록 (PRD F-HIST-06)
 */
export function BookmarkList() {
  const { bookmarks, removeBookmark } = useSidebarStore()

  if (bookmarks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center text-sm text-muted-foreground">
        <Star className="h-8 w-8 opacity-30" />
        <p>저장된 답변이 없습니다.</p>
        <p className="text-xs">답변의 ★ 아이콘을 눌러 보관하세요.</p>
      </div>
    )
  }

  return (
    <ul className="px-2 py-2 space-y-2">
      {bookmarks.map((bm) => (
        <li
          key={bm.messageId}
          className="group relative rounded-md border bg-card p-3 text-sm"
        >
          <p className="line-clamp-3 text-card-foreground">{bm.content}</p>
          <p className="mt-1 text-[10px] text-muted-foreground">
            {formatDateTime(bm.bookmarkedAt)}
          </p>

          {/* 즐겨찾기 해제 */}
          <button
            onClick={() => removeBookmark(bm.messageId)}
            aria-label="즐겨찾기 해제"
            className="absolute right-2 top-2 hidden text-knu-accent group-hover:block"
          >
            <Star className="h-3.5 w-3.5 fill-current" />
          </button>
        </li>
      ))}
    </ul>
  )
}
