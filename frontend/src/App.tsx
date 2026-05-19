import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppLayout } from '@/components/layout/AppLayout'
import ChatPage from '@/pages/ChatPage'
import NotFoundPage from '@/pages/NotFoundPage'

/**
 * 라우팅 루트 컴포넌트
 * /       → ChatPage (AppLayout 내부)
 * /*      → 404 NotFoundPage
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<ChatPage />} />
        </Route>
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </BrowserRouter>
  )
}
