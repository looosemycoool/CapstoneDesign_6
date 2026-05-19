import { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { Header } from './Header'
import { Sidebar } from '@/components/sidebar/Sidebar'
import { useSidebarStore } from '@/stores'
import { useChatHistory } from '@/hooks'
import { cn } from '@/lib/utils'

/**
 * 전체 레이아웃 — 사이드바(280px) + 콘텐츠 영역 (PRD 5. IA)
 * 모바일(< md): 사이드바 오버레이 방식
 * 데스크톱: 사이드바 고정 슬라이드
 */
export function AppLayout() {
  const { isOpen, setSidebarOpen } = useSidebarStore()
  const { loadHistory } = useChatHistory()

  // 마운트 시 히스토리 로드
  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* 모바일 딤 배경 */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* 사이드바 */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-30 w-[280px] transform transition-transform duration-200 ease-in-out',
          'md:relative md:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <Sidebar />
      </aside>

      {/* 메인 영역 */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
