import { supabase } from '@/lib/supabase'

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || '').replace(/\/$/, '')

export interface User {
  id: string
  name: string
  email: string
  avatar?: string
}

export interface Material {
  id: string
  title: string
  source_type: 'pdf' | 'url' | 'topic'
  topic?: string
  status: 'pending' | 'processing' | 'ready' | 'error' | 'failed'
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

export interface QuizQuestion {
  id: string
  type: 'mcq' | 'true_false'
  question: string
  options?: string[]
  correct_answer: string
}

export interface QuizResult {
  total: number
  correct: number
  incorrect: number
  score: number
  answers: {
    question_id: string
    user_answer: string
    correct_answer: string
    is_correct: boolean
  }[]
}

export interface ChatSession {
  id: string
  title: string
  material_id: string
  user_id: string
  created_at: string
  updated_at: string
}

function buildHeaders(token: string | null): HeadersInit {
  const hfToken = process.env.NEXT_PUBLIC_HF_TOKEN
  return {
    'Content-Type': 'application/json',
    // HF token for private space access goes in standard Authorization header
    ...(hfToken && { Authorization: `Bearer ${hfToken}` }),
    // User JWT goes in custom header if HF token is present, otherwise fallback to Authorization
    ...(!hfToken && token && { Authorization: `Bearer ${token}` }),
    ...(token && { 'X-Auth-Token': token }),
  }
}

// Force a full sign-out when the refresh token is expired or invalid.
function forceSignOut() {
  if (typeof window === 'undefined') return
  localStorage.removeItem('auth_token')
  localStorage.removeItem('auth_user')
  localStorage.removeItem('usage_cache')
  localStorage.removeItem('cached_materials')
  supabase.auth.signOut().catch(() => {})
}

/** Attempt to refresh the Supabase session and return the new access token, or null. */
async function refreshToken(): Promise<string | null> {
  try {
    const { data, error } = await supabase.auth.refreshSession()
    if (error || !data.session) return null
    const newToken = data.session.access_token
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', newToken)
    }
    return newToken
  } catch {
    return null
  }
}

async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: { ...buildHeaders(token), ...options.headers },
  })

  // ── Token-refresh retry ───────────────────────────────────────────────────
  // If the server returns 401, the stored JWT is likely expired (e.g. the user
  // left the tab open overnight).  Ask Supabase to refresh the session and retry
  // the request once with the new token before surfacing the error.
  // If the refresh itself fails (refresh token also expired), force a full sign-out.
  if (response.status === 401) {
    const freshToken = await refreshToken()
    if (freshToken) {
      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: { ...buildHeaders(freshToken), ...options.headers },
      })
      if (retryResponse.status === 401) {
        // Refresh token was also invalid — session is fully dead, force logout.
        forceSignOut()
        throw new Error('Session expired. Please sign in again.')
      }
      if (!retryResponse.ok) {
        const error = await retryResponse.json().catch(() => ({ message: 'An error occurred' }))
        throw new Error(error.message || error.detail || `HTTP error! status: ${retryResponse.status}`)
      }
      return retryResponse.json()
    }
    // refreshToken() returned null — refresh token missing or Supabase unreachable.
    forceSignOut()
    throw new Error('Session expired. Please sign in again.')
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'An error occurred' }))
    throw new Error(error.message || error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

export const authAPI = {
  googleAuth: (token: string) =>
    fetchAPI<{ token: string; user: User }>('/api/auth/google', {
      method: 'POST',
      body: JSON.stringify({ token }),
    }),

  deleteAccount: () =>
    fetchAPI<{ status: string; message: string }>('/api/auth/me', {
      method: 'DELETE',
    }),

  getProfile: () =>
    fetchAPI<{ status: string; user: User }>('/api/auth/profile'),

  updateProfile: (data: { name?: string; avatar_url?: string }) =>
    fetchAPI<{ status: string; user: User }>('/api/auth/profile', {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
}

export const materialsAPI = {
  list: () => fetchAPI<Material[]>('/api/materials'),

  uploadPDF: async (file: File): Promise<{ material_id: string; title: string; chunks_count: number }> => {
    let token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    const formData = new FormData()
    formData.append('file', file)

    // buildHeaders omits 'Content-Type' for FormData so the browser can set the boundary
    const makeHeaders = (t: string | null) => {
      const hfToken = process.env.NEXT_PUBLIC_HF_TOKEN
      return {
        ...(hfToken && { Authorization: `Bearer ${hfToken}` }),
        ...(!hfToken && t && { Authorization: `Bearer ${t}` }),
        ...(t && { 'X-Auth-Token': t }),
      }
    }

    let response = await fetch(`${API_BASE_URL}/api/materials/upload-pdf`, {
      method: 'POST',
      headers: makeHeaders(token),
      body: formData,
    })

    // Token-refresh retry (same logic as fetchAPI)
    if (response.status === 401) {
      const freshToken = await refreshToken()
      if (freshToken) {
        token = freshToken
        response = await fetch(`${API_BASE_URL}/api/materials/upload-pdf`, {
          method: 'POST',
          headers: makeHeaders(token),
          body: formData,
        })
        if (response.status === 401) {
          forceSignOut()
          throw new Error('Session expired. Please sign in again.')
        }
      } else {
        forceSignOut()
        throw new Error('Session expired. Please sign in again.')
      }
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Failed to upload PDF' }))
      throw new Error(error.message || error.detail || 'Failed to upload PDF')
    }

    return response.json()
  },

  scrapeURL: (url: string) =>
    fetchAPI<{ material_id: string; title: string; chunks_count: number }>('/api/materials/scrape-url', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),

  addTopic: (topic: string) =>
    fetchAPI<{ material_id: string; title: string }>('/api/materials/topic', {
      method: 'POST',
      body: JSON.stringify({ topic }),
    }),

  search: (q: string) =>
    fetchAPI<{ results: { material_id: string; relevance: number }[] }>('/api/materials/search', {
      method: 'POST',
      body: JSON.stringify({ q }),
    }),

  summarize: (material_id: string) =>
    fetchAPI<{ summary: string; time_taken: number }>('/api/materials/summarize', {
      method: 'POST',
      body: JSON.stringify({ material_id }),
    }),

  get: (material_id: string) =>
    fetchAPI<Material>(`/api/materials/${material_id}`),

  getSummary: (material_id: string) =>
    fetchAPI<{ summary: string; time_taken: number }>(`/api/materials/${material_id}/summary`),

  delete: (material_id: string) =>
    fetchAPI<{ status: string }>(`/api/materials/${material_id}`, { method: 'DELETE' }),

  rename: (material_id: string, title: string) =>
    fetchAPI<{ status: string }>(`/api/materials/${material_id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  bulkDelete: (material_ids: string[]) =>
    fetchAPI<{ status: string }>('/api/materials/bulk-delete', {
      method: 'POST',
      body: JSON.stringify({ material_ids }),
    }),
}

export const tutorAPI = {
  ask: (query: string, source_type: string, material_id?: string, session_id?: string) =>
    fetchAPI<{ answer: string; source: string; time_taken: number; memory_id: string }>('/api/tutor/ask', {
      method: 'POST',
      body: JSON.stringify({ query, source_type, material_id, session_id }),
    }),

  // Session management
  listSessions: (material_id: string) =>
    fetchAPI<ChatSession[]>(`/api/tutor/sessions?material_id=${material_id}`),

  createSession: (material_id: string, title?: string) =>
    fetchAPI<ChatSession>('/api/tutor/sessions', {
      method: 'POST',
      body: JSON.stringify({ material_id, title }),
    }),

  deleteSession: (session_id: string) =>
    fetchAPI<{ status: string }>(`/api/tutor/sessions/${session_id}`, { method: 'DELETE' }),

  renameSession: (session_id: string, title: string) =>
    fetchAPI<{ status: string }>(`/api/tutor/sessions/${session_id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  extractTitle: (session_id: string, query: string) =>
    fetchAPI<{ status: string; title: string }>(`/api/tutor/sessions/${session_id}/extract-title`, {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  getSessionMessages: (session_id: string) =>
    fetchAPI<{ id: string; role: string; content: string; created_at: string }[]>(
      `/api/tutor/sessions/${session_id}/messages`
    ),

  // Legacy save/load
  saveChat: (material_id: string, messages: ChatMessage[]) =>
    fetchAPI<{ status: string }>('/api/tutor/chat/save', {
      method: 'POST',
      body: JSON.stringify({ material_id, messages }),
    }),

  loadChat: (material_id: string) =>
    fetchAPI<{ messages: ChatMessage[] }>(`/api/tutor/chat/${material_id}`),
}

export const quizAPI = {
  generate: (
    difficulty: string,
    mcq_count: number,
    tf_count: number,
    source_type: string,
    material_id?: string,
    topic?: string,
  ) =>
    fetchAPI<{ quiz: Record<string, unknown>; quiz_id: string }>('/api/quiz/generate', {
      method: 'POST',
      body: JSON.stringify({ difficulty, mcq_count, tf_count, source_type, material_id, topic }),
    }),

  list: (material_id?: string) => {
    const params = material_id ? `?material_id=${material_id}` : ''
    return fetchAPI<any[]>(`/api/quiz/list${params}`)
  },

  saveResult: (quiz_id: string, result_data: Record<string, unknown>) =>
    fetchAPI<{ status: string }>('/api/quiz/save-result', {
      method: 'POST',
      body: JSON.stringify({ quiz_id, result_data }),
    }),

  getResults: (quiz_id: string) =>
    fetchAPI<any[]>(`/api/quiz/results/${quiz_id}`),
}



export const usageAPI = {
  getUsage: () => fetchAPI<{ used: number; limit: number; remaining: number }>('/api/usage'),
}
