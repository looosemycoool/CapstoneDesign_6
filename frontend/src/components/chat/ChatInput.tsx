import { useRef, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { ArrowUp, Square } from 'lucide-react'
import { useChatStore } from '@/stores'
import { useSendMessage } from '@/hooks'
import { cn } from '@/lib/utils'

interface FormValues {
  message: string
}

/**
 * 질의 입력창 (PRD F-INPUT-01 ~ F-INPUT-04)
 * - Enter → 전송 / Shift+Enter → 줄바꿈
 * - 자동 높이 조절 (1행~5행)
 * - 빈 상태 전송 방지
 * - 스트리밍 중 [■ 중단] 버튼 전환
 */
export function ChatInput() {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isStreaming } = useChatStore()
  const { sendMessage, abortStream } = useSendMessage()

  const { register, handleSubmit, watch, reset, setValue } = useForm<FormValues>({
    defaultValues: { message: '' },
  })

  const messageValue = watch('message')
  const isEmpty = !messageValue.trim()

  // 자동 높이 조절 (PRD F-INPUT-03)
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    const lineHeight = 24
    const maxHeight = lineHeight * 5
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [messageValue])

  // 페이지 진입 시 자동 포커스 (PRD 접근성)
  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  const onSubmit = async ({ message }: FormValues) => {
    if (isEmpty || isStreaming) return
    reset()
    await sendMessage(message)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(onSubmit)()
    }
  }

  // react-hook-form ref 병합
  const { ref: rhfRef, ...rest } = register('message')

  return (
    <div className="border-t bg-background px-4 py-3">
      <form
        onSubmit={handleSubmit(onSubmit)}
        className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border bg-card px-4 py-2 shadow-sm focus-within:ring-2 focus-within:ring-ring"
      >
        <textarea
          {...rest}
          ref={(el) => {
            rhfRef(el)
            ;(textareaRef as React.MutableRefObject<HTMLTextAreaElement | null>).current = el
          }}
          onKeyDown={handleKeyDown}
          placeholder="KNU 공지사항에 대해 질문하세요..."
          rows={1}
          aria-label="메시지 입력"
          className="flex-1 resize-none bg-transparent py-1 text-sm outline-none placeholder:text-muted-foreground"
          style={{ minHeight: '24px', maxHeight: `${24 * 5}px` }}
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={abortStream}
            aria-label="답변 생성 중단"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-foreground text-background hover:opacity-80 transition-opacity"
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={isEmpty}
            aria-label="메시지 전송"
            className={cn(
              'flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors',
              isEmpty
                ? 'bg-muted text-muted-foreground cursor-not-allowed'
                : 'bg-knu text-white hover:bg-knu-light',
            )}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        )}
      </form>

      <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
        Enter로 전송 · Shift+Enter로 줄바꿈
      </p>
    </div>
  )
}
