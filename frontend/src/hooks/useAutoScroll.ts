import { useRef, useCallback, useEffect } from 'react'
import { AUTO_SCROLL_THRESHOLD_PX } from '@/constants'

/**
 * 스트리밍 중 자동 스크롤 + 사용자 스크롤 감지 (PRD F-UX-01, F-UX-02)
 *
 * @returns containerRef  스크롤 컨테이너에 부착할 ref
 * @returns scrollToBottom 수동으로 최하단으로 스크롤
 */
export function useAutoScroll(isStreaming: boolean) {
  const containerRef = useRef<HTMLDivElement>(null)
  const userScrolledUp = useRef(false)

  const isNearBottom = useCallback(() => {
    const el = containerRef.current
    if (!el) return true
    return el.scrollHeight - el.scrollTop - el.clientHeight <= AUTO_SCROLL_THRESHOLD_PX
  }, [])

  const scrollToBottom = useCallback(() => {
    containerRef.current?.scrollTo({
      top: containerRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [])

  // 사용자 스크롤 감지
  const handleScroll = useCallback(() => {
    userScrolledUp.current = !isNearBottom()
  }, [isNearBottom])

  // 스트리밍 중 청크 누적마다 최하단으로 스크롤
  useEffect(() => {
    if (isStreaming && !userScrolledUp.current) {
      containerRef.current?.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'instant',
      })
    }
  })

  // 스트리밍 완료 시 자동 스크롤 재개 플래그 초기화
  useEffect(() => {
    if (!isStreaming) {
      userScrolledUp.current = false
    }
  }, [isStreaming])

  return { containerRef, scrollToBottom, handleScroll }
}
