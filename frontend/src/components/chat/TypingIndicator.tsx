import { Bot } from 'lucide-react'

/**
 * 어시스턴트 응답 대기 중 표시 (PRD F-CHAT-03)
 * 3개의 점이 순차적으로 opacity 변화하는 애니메이션
 */
export function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 py-2" role="status" aria-label="답변 생성 중">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-knu">
        <Bot className="h-4 w-4 text-white" />
      </div>
      <div className="flex items-center gap-1 rounded-2xl border bg-card px-4 py-3">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="typing-dot h-2 w-2 rounded-full bg-muted-foreground animate-typing-dot"
            style={{ animationDelay: `${i * 0.2}s` }}
            aria-hidden="true"
          />
        ))}
      </div>
    </div>
  )
}
