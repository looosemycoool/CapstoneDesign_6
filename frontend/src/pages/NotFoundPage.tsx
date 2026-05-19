import { Link } from 'react-router-dom'
import { Megaphone } from 'lucide-react'

/** 404 페이지 (PRD 5. IA) */
export default function NotFoundPage() {
  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 text-center">
      <Megaphone className="h-16 w-16 text-muted-foreground opacity-30" />
      <div>
        <h1 className="text-4xl font-bold">404</h1>
        <p className="mt-2 text-muted-foreground">페이지를 찾을 수 없습니다.</p>
      </div>
      <Link
        to="/"
        className="rounded-lg bg-knu px-4 py-2 text-sm font-medium text-white hover:bg-knu-light transition-colors"
      >
        홈으로 돌아가기
      </Link>
    </div>
  )
}
