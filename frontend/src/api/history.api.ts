import axiosInstance from './axiosInstance'
import type { HistoryListResponse, SessionDetailResponse } from '@/types'

/** GET /api/history — 대화 목록 조회 (PRD 9-2) */
export async function fetchHistory(): Promise<HistoryListResponse> {
  const { data } = await axiosInstance.get<HistoryListResponse>('/history')
  return data
}

/** GET /api/history/:sessionId — 특정 세션 메시지 조회 (PRD 9-3) */
export async function fetchSessionDetail(
  sessionId: string,
): Promise<SessionDetailResponse> {
  const { data } = await axiosInstance.get<SessionDetailResponse>(
    `/history/${sessionId}`,
  )
  return data
}

/** DELETE /api/history/:sessionId — 대화 삭제 (PRD 9-4) */
export async function deleteSession(sessionId: string): Promise<void> {
  await axiosInstance.delete(`/history/${sessionId}`)
}
