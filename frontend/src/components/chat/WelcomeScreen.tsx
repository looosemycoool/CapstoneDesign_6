import { Megaphone } from 'lucide-react'
import { EXAMPLE_QUESTIONS } from '@/constants'
import { useSendMessage } from '@/hooks'

/**
 * 웰컴 화면 — 메시지가 없을 때 표시 (PRD F-INIT-03)
 * 추천 질문 카드 4개 표시, 클릭 시 즉시 전송
 */
export function WelcomeScreen() {
  const { sendMessage } = useSendMessage()

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-4">
      {/* 타이틀 */}
      <div className="flex flex-col items-center gap-3 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-knu">
          <Megaphone className="h-8 w-8 text-white" />
        </div>
        <h1 className="text-2xl font-bold text-foreground">무엇을 도와드릴까요?</h1>
        <p className="text-sm text-muted-foreground">
          KNU 공지사항에 대해 자연어로 질문해보세요.
        </p>
      </div>

      {/* 추천 질문 카드 */}
      <div className="grid w-full max-w-xl grid-cols-1 gap-3 sm:grid-cols-2">
        {EXAMPLE_QUESTIONS.map(({ emoji, text }) => (
          <button
            key={text}
            onClick={() => sendMessage(text)}
            className="flex items-start gap-3 rounded-xl border bg-card p-4 text-left text-sm transition-colors hover:bg-accent hover:border-knu focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span className="text-xl" aria-hidden="true">{emoji}</span>
            <span className="text-card-foreground">{text}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
