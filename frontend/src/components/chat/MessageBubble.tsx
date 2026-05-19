import { useState } from 'react'
import { Bot, Copy, Check, Star, RefreshCw, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import { SourceAccordion } from './SourceAccordion'
import { useChatStore, useSidebarStore } from '@/stores'
import { stripMarkdown } from '@/lib/utils'
import { COPY_FEEDBACK_MS } from '@/constants'
import { cn } from '@/lib/utils'
import type { Message } from '@/types'

interface Props {
  message: Message
  onRetry?: () => void
}

/**
 * 메시지 말풍선 (PRD F-CHAT-01, F-CHAT-02, F-CHAT-04, F-UX-05)
 * - user: 우측 정렬
 * - assistant: 좌측 정렬 + 액션 아이콘 hover 노출
 */
export function MessageBubble({ message, onRetry }: Props) {
  const [copied, setCopied] = useState(false)
  const { toggleBookmark, sessionId } = useChatStore()
  const { addBookmark, removeBookmark, isBookmarked } = useSidebarStore()

  const bookmarked = isBookmarked(message.id)
  const isUser = message.role === 'user'

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(stripMarkdown(message.content))
      setCopied(true)
      setTimeout(() => setCopied(false), COPY_FEEDBACK_MS)
    } catch {
      // clipboard API 미지원 환경 무시
    }
  }

  const handleBookmark = () => {
    if (bookmarked) {
      removeBookmark(message.id)
      toggleBookmark(message.id)
    } else {
      addBookmark(message, sessionId ?? '')
      toggleBookmark(message.id)
    }
  }

  if (isUser) {
    return (
      <div className="flex justify-end py-2">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-knu px-4 py-3 text-sm text-white">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="group flex items-start gap-3 py-2">
      {/* 봇 아이콘 */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-knu">
        <Bot className="h-4 w-4 text-white" />
      </div>

      <div className="flex-1 min-w-0">
        {/* 에러 버블 */}
        {message.isError ? (
          <div className="flex items-start gap-2 rounded-2xl rounded-tl-sm border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <p>연결이 원활하지 않습니다. 잠시 후 다시 시도해주세요.</p>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="mt-2 flex items-center gap-1 text-xs underline underline-offset-2"
                >
                  <RefreshCw className="h-3 w-3" />
                  다시 시도
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-2xl rounded-tl-sm border bg-card px-4 py-3">
            {/* 마크다운 렌더링 (PRD F-CHAT-02) */}
            <div
              className="prose prose-sm max-w-none dark:prose-invert"
              aria-live="polite"
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeSanitize]}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* 출처 아코디언 */}
            {message.sources && message.sources.length > 0 && (
              <SourceAccordion sources={message.sources} />
            )}
          </div>
        )}

        {/* 액션 아이콘 — hover 시 노출 */}
        {!message.isError && (
          <div className="mt-1 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
            <button
              onClick={handleCopy}
              aria-label={copied ? '복사됨' : '답변 복사'}
              className="rounded p-1 text-muted-foreground hover:text-foreground transition-colors"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
            </button>

            <button
              onClick={handleBookmark}
              aria-label={bookmarked ? '즐겨찾기 해제' : '즐겨찾기 추가'}
              className={cn(
                'rounded p-1 transition-colors',
                bookmarked
                  ? 'text-knu-accent'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <Star className={cn('h-3.5 w-3.5', bookmarked && 'fill-current')} />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
