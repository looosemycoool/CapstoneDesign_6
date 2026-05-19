import { useState } from 'react'
import { ChevronDown, Paperclip, ExternalLink } from 'lucide-react'
import type { AnnouncementSource } from '@/types'
import { MAX_SOURCE_CARDS } from '@/constants'
import { cn } from '@/lib/utils'

interface Props {
  sources: AnnouncementSource[]
}

/**
 * 출처 아코디언 컴포넌트 (PRD F-SRC-01, F-SRC-02, F-SRC-03)
 * 기본 접힘 상태, 클릭 시 출처 카드 목록 펼침
 */
export function SourceAccordion({ sources }: Props) {
  const [isOpen, setIsOpen] = useState(false)

  if (!sources || sources.length === 0) return null

  const visible = sources.slice(0, MAX_SOURCE_CARDS)

  return (
    <div className="mt-2">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
        aria-controls="source-list"
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Paperclip className="h-3.5 w-3.5" />
        <span>참조 출처 보기 ({visible.length})</span>
        <ChevronDown
          className={cn(
            'h-3.5 w-3.5 transition-transform duration-200',
            isOpen && 'rotate-180',
          )}
        />
      </button>

      {isOpen && (
        <ul id="source-list" className="mt-2 space-y-2">
          {visible.map((src) => (
            <li key={src.id}>
              <a
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex flex-col gap-1 rounded-lg border bg-card p-3 hover:border-knu transition-colors"
              >
                {/* 카테고리 배지 + 제목 */}
                <div className="flex items-center gap-2">
                  <span className="shrink-0 rounded-full bg-knu/10 px-2 py-0.5 text-[10px] font-semibold text-knu">
                    {src.category}
                  </span>
                  <span className="truncate text-xs font-medium text-card-foreground group-hover:text-knu">
                    {src.title}
                  </span>
                  <ExternalLink className="ml-auto h-3 w-3 shrink-0 text-muted-foreground" />
                </div>

                {/* 내용 첫 문장 */}
                {src.summary && (
                  <p className="line-clamp-2 text-[11px] text-muted-foreground">
                    {src.summary}
                  </p>
                )}

                {/* 작성일 */}
                <p className="text-[10px] text-muted-foreground">{src.publishedAt}</p>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
