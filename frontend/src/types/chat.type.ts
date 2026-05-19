import type { AnnouncementSource } from './message.type'

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messageCount: number
}

export interface BookmarkedMessage {
  messageId: string
  sessionId: string
  content: string
  sources?: AnnouncementSource[]
  bookmarkedAt: string
}

// SSE 스트림 청크 타입
export type StreamChunkType = 'text' | 'sources' | 'session' | 'done' | 'error'

export interface StreamChunk {
  type: StreamChunkType
  content?: string
  sources?: AnnouncementSource[]
  sessionId?: string
  error?: string
}

// API 요청/응답 타입
export interface ChatRequest {
  sessionId: string | null
  message: string
}

export interface HistoryListResponse {
  sessions: ChatSession[]
}

export interface SessionDetailResponse {
  sessionId: string
  messages: import('./message.type').Message[]
}

export interface MetaResponse {
  lastCrawledAt: string    // ISO 8601
  totalAnnouncements: number
}
