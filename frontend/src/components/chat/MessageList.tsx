import { useChatStore } from '@/stores'
import { useSendMessage, useAutoScroll } from '@/hooks'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'
import { WelcomeScreen } from './WelcomeScreen'

/**
 * 메시지 목록 + 자동 스크롤 (PRD F-UX-01, F-UX-02, F-UX-07)
 */
export function MessageList() {
  const { messages, isStreaming, lastUserMessage } = useChatStore()
  const { sendMessage } = useSendMessage()
  const { containerRef, handleScroll } = useAutoScroll(isStreaming)

  const showTypingIndicator =
    isStreaming &&
    (messages.length === 0 ||
      messages[messages.length - 1]?.role !== 'assistant' ||
      messages[messages.length - 1]?.content === '')

  if (messages.length === 0) {
    return <WelcomeScreen />
  }

  const handleRetry = () => {
    if (lastUserMessage) {
      sendMessage(lastUserMessage)
    }
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto scrollbar-thin px-4 py-4 space-y-0"
    >
      <div className="mx-auto max-w-3xl">
        {messages.map((msg, idx) => {
          const isLast = idx === messages.length - 1
          return (
            <MessageBubble
              key={msg.id}
              message={msg}
              onRetry={msg.isError && isLast ? handleRetry : undefined}
            />
          )
        })}

        {showTypingIndicator && <TypingIndicator />}
      </div>
    </div>
  )
}
