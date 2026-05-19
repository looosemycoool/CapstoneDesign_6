import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'

/**
 * 메인 채팅 페이지 — 메시지 목록 + 입력창 (PRD 5. IA)
 * 라우트: /
 */
export default function ChatPage() {
  return (
    <div className="flex h-full flex-col">
      <MessageList />
      <ChatInput />
    </div>
  )
}
