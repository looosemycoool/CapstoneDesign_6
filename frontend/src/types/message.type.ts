export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  sources?: AnnouncementSource[]
  createdAt: string
  isError?: boolean
  isBookmarked?: boolean
}

export interface AnnouncementSource {
  id: string
  title: string
  category: string          // "학사", "장학", "취업" 등
  summary: string           // 내용의 첫 문장
  publishedAt: string       // "YYYY-MM-DD"
  url: string
}
