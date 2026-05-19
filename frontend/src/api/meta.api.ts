import axiosInstance from './axiosInstance'
import type { MetaResponse } from '@/types'

/** GET /api/meta — 마지막 크롤링 시각 등 메타 정보 (PRD 9-5, F-INIT-02) */
export async function fetchMeta(): Promise<MetaResponse> {
  const { data } = await axiosInstance.get<MetaResponse>('/meta')
  return data
}
