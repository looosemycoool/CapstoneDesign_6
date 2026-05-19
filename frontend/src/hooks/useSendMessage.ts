import { useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { sendChatMessage } from '@/api'
import { useChatStore } from '@/stores'
import { replaceSlang } from '@/constants'
import type { Message, StreamChunk } from '@/types'

/**
 * 메시지 전송 + SSE 스트리밍 수신 + Abort 처리를 담당하는 훅 (PRD 8-1)
 */
export function useSendMessage() {
  const {
    sessionId,
    addMessage,
    appendStreamChunk,
    finalizeStream,
    setStreaming,
    setAbortController,
    setSessionId,
    setLastUserMessage,
    setMessageError,
  } = useChatStore()

  const sendMessage = useCallback(
    async (text: string) => {
      const normalized = text.trim()
      if (!normalized) return

      // 줄임말 치환 (PRD F-UX-06)
      const replaced = replaceSlang(normalized)

      // 사용자 메시지 추가
      const userMsg: Message = {
        id: uuidv4(),
        role: 'user',
        content: normalized, // 화면에는 원문 표시
        createdAt: new Date().toISOString(),
      }
      addMessage(userMsg)
      setLastUserMessage(normalized)

      // 어시스턴트 플레이스홀더 추가
      const assistantId = uuidv4()
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
      }
      addMessage(assistantMsg)
      setStreaming(true)

      const ctrl = new AbortController()
      setAbortController(ctrl)

      try {
        await sendChatMessage(
          { sessionId, message: replaced },
          (chunk: StreamChunk) => {
            if (chunk.type === 'text' && chunk.content) {
              appendStreamChunk(chunk.content)
            } else if (chunk.type === 'session' && chunk.sessionId) {
              setSessionId(chunk.sessionId)
            } else if (chunk.type === 'sources') {
              finalizeStream(chunk.sources)
            } else if (chunk.type === 'done') {
              finalizeStream()
            } else if (chunk.type === 'error') {
              setMessageError(assistantId)
            }
          },
          ctrl.signal,
        )
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') {
          // 사용자가 명시적으로 중단 — 현재까지 텍스트 유지
          setStreaming(false)
          setAbortController(null)
        } else {
          setMessageError(assistantId)
        }
      }
    },
    [
      sessionId,
      addMessage,
      appendStreamChunk,
      finalizeStream,
      setStreaming,
      setAbortController,
      setSessionId,
      setLastUserMessage,
      setMessageError,
    ],
  )

  const abortStream = useCallback(() => {
    useChatStore.getState().abortController?.abort()
  }, [])

  return { sendMessage, abortStream }
}
