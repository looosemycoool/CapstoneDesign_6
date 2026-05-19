import type { StreamChunk, ChatRequest } from '@/types'
import { STREAM_TIMEOUT_MS } from '@/constants'

/**
 * POST /api/chat — SSE 스트리밍 챗봇 응답 수신 (PRD 9-1)
 *
 * @param request   sessionId + message
 * @param onChunk   청크 수신 콜백
 * @param signal    AbortController.signal (중단 버튼용)
 */
export async function sendChatMessage(
  request: ChatRequest,
  onChunk: (chunk: StreamChunk) => void,
  signal: AbortSignal,
): Promise<void> {
  const timeoutId = setTimeout(() => {
    // 30초 타임아웃 처리 — AbortController가 외부에서 이미 abort 될 수도 있음
  }, STREAM_TIMEOUT_MS)

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('ReadableStream not supported')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const raw = line.slice(6).trim()
        if (!raw) continue

        try {
          const chunk = JSON.parse(raw) as StreamChunk
          onChunk(chunk)
          if (chunk.type === 'done' || chunk.type === 'error') return
        } catch {
          // JSON 파싱 오류는 무시 (불완전한 청크)
        }
      }
    }
  } finally {
    clearTimeout(timeoutId)
  }
}
