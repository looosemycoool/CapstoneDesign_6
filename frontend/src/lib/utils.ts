import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/** shadcn/ui 스타일 유틸 — cn('base', condition && 'extra') 패턴으로 사용 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** ISO 8601 날짜 문자열을 "YYYY.MM.DD. HH:MM" 형식으로 변환 */
export function formatDateTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}. ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/** 날짜 그룹 레이블 반환 ("오늘" | "어제" | "이전 7일" | "더 오래된 대화") */
export function getDateGroup(iso: string): string {
  const now  = new Date()
  const date = new Date(iso)
  const diffDays = Math.floor((now.getTime() - date.getTime()) / 86_400_000)
  if (diffDays === 0) return '오늘'
  if (diffDays === 1) return '어제'
  if (diffDays <= 7)  return '이전 7일'
  return '더 오래된 대화'
}

/** 마크다운 태그 제거 후 순수 텍스트 반환 (클립보드 복사용) */
export function stripMarkdown(md: string): string {
  return md
    .replace(/#{1,6}\s/g, '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/_(.+?)_/g, '$1')
    .replace(/`{1,3}[^`]*`{1,3}/g, '')
    .replace(/\[(.+?)\]\(.+?\)/g, '$1')
    .replace(/!\[.+?\]\(.+?\)/g, '')
    .replace(/^\s*[-*+]\s/gm, '')
    .replace(/^\s*\d+\.\s/gm, '')
    .trim()
}
