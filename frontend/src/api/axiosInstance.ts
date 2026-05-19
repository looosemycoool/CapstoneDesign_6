import axios from 'axios'

/**
 * 공통 axios 인스턴스.
 * baseURL은 vite.config.ts proxy 설정으로 /api → http://localhost:8000/api 로 전달.
 */
const axiosInstance = axios.create({
  baseURL: '/api',
  timeout: 10_000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 요청 인터셉터 — 공통 헤더 등 추가 가능
axiosInstance.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error),
)

// 응답 인터셉터 — 공통 에러 처리
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    // 네트워크 오류 / 타임아웃 등 axios 레벨 에러
    return Promise.reject(error)
  },
)

export default axiosInstance
