import { useEffect, useState } from 'react'
import { Menu, Megaphone } from 'lucide-react'
import { useSidebarStore } from '@/stores'
import { fetchMeta } from '@/api'
import { formatDateTime } from '@/lib/utils'

/**
 * 헤더 컴포넌트 (PRD F-INIT-01, F-INIT-02, F-HIST-01)
 * - 로고 + 서비스 타이틀
 * - 마지막 업데이트 시각
 * - 햄버거 사이드바 토글
 */
export function Header() {
  const { toggleSidebar } = useSidebarStore()
  const [lastCrawledAt, setLastCrawledAt] = useState<string | null>(null)

  useEffect(() => {
    fetchMeta()
      .then((meta) => setLastCrawledAt(meta.lastCrawledAt))
      .catch(() => { /* 실패 시 노출 생략 */ })
  }, [])

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b bg-background px-4">
      {/* 좌측: 햄버거 + 로고 */}
      <div className="flex items-center gap-3">
        <button
          onClick={toggleSidebar}
          aria-label="사이드바 열기/닫기"
          className="rounded-md p-1.5 hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="flex items-center gap-2">
          <Megaphone className="h-5 w-5 text-knu" />
          <div className="leading-tight">
            <p className="text-sm font-bold text-knu">Lucid</p>
            <p className="hidden text-[10px] text-muted-foreground sm:block">
              KNU 공지사항 챗봇
            </p>
          </div>
        </div>
      </div>

      {/* 우측: 마지막 업데이트 시각 */}
      {lastCrawledAt && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          마지막 업데이트: {formatDateTime(lastCrawledAt)}
        </p>
      )}
    </header>
  )
}
