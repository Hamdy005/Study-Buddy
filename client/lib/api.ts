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

// ── In-memory token store ─────────────────────────────────────────────────────
// The access JWT never touches localStorage — it lives only in this module-level
// variable (and in React state in AuthProvider). On a hard page refresh it is
// cleared and the AuthProvider silently re-issues it via the refresh cookie.

let _accessToken: string | null = null

export function setApiToken(token: string | null): void {
  _accessToken = token
}

export function getApiToken(): string | null {
  return _accessToken
}

// ── Header builder ────────────────────────────────────────────────────────────

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

let _forceSignOutListeners: Array<() => void> = []

export function onForceSignOut(cb: () => void): () => void {
  _forceSignOutListeners.push(cb)
  return () => {
    _forceSignOutListeners = _forceSignOutListeners.filter(fn => fn !== cb)
  }
}

function forceSignOut() {
  if (typeof window === 'undefined') return
  if (_forceSignOutTimer) return // already scheduled
  _forceSignOutTimer = setTimeout(async () => {
    _forceSignOutTimer = null
    // Revoke the refresh token server-side (best-effort)
    try {
      await fetch(`${API_BASE_URL}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include', // sends the HttpOnly cookie
      })
    } catch { /* ignore network errors during force logout */ }
    // Clear in-memory token
    setApiToken(null)
    // Clear local state
    localStorage.removeItem('auth_user')
    localStorage.removeItem('usage_cache')
    localStorage.removeItem('cached_materials')
    
    // Notify listeners (e.g. AuthContext)
    _forceSignOutListeners.forEach(fn => fn())

    // Sign out of Supabase as well (for Google OAuth sessions)
    await supabase.auth.signOut().catch(() => {})

    // Redirect to login page
    if (window.location.pathname !== '/') {
      window.location.href = '/'
    }
  }, 100)
}

// ── Refresh lock ──────────────────────────────────────────────────────────────
// Multiple concurrent API calls can each receive 401 at the same time.
// This lock ensures only ONE refresh attempt runs; all others wait for the
// same promise result.

let _refreshPromise: Promise<string | null> | null = null

/**
 * Call POST /api/auth/refresh.
 * The browser automatically sends the HttpOnly refresh-token cookie.
 * On success, updates the in-memory token store and returns the new JWT.
 * On failure, returns null (caller should forceSignOut).
 */
async function refreshToken(): Promise<string | null> {
  if (_refreshPromise) return _refreshPromise

  _refreshPromise = (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
        method: 'POST',
        credentials: 'include', // sends the HttpOnly refresh cookie
      })
      if (!res.ok) return null
      const data = await res.json()
      const newToken: string | null = data.access_token ?? null
      if (newToken) setApiToken(newToken)
      return newToken
    } catch {
      return null
    } finally {
      // Clear lock after a short delay so closely-spaced calls share the result,
      // but future calls (e.g. minutes later) get a fresh attempt.
      setTimeout(() => { _refreshPromise = null }, 2000)
    }
  })()

  return _refreshPromise
}

// ── Core fetch wrapper ────────────────────────────────────────────────────────

async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getApiToken()

  // Don't fire a network request without a token — let the caller handle it.
  if (!token) {
    throw new Error('Not authenticated')
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    credentials: 'include', // always include cookies (needed for refresh endpoint)
    headers: { ...buildHeaders(token), ...options.headers },
  })

  // ── Token-refresh retry ───────────────────────────────────────────────────
  // If the server returns 401 the access token has likely expired.
  // Call our /api/auth/refresh (cookie sent automatically) and retry once.
  if (response.status === 401) {
    const freshToken = await refreshToken()
    if (freshToken) {
      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        credentials: 'include',
        headers: { ...buildHeaders(freshToken), ...options.headers },
      })
      if (retryResponse.status === 401) {
        forceSignOut()
        throw new Error('Session expired. Please sign in again.')
      }
      if (!retryResponse.ok) {
        const error = await retryResponse.json().catch(() => ({ message: 'An error occurred' }))
        throw new Error(error.message || error.detail || `HTTP error! status: ${retryResponse.status}`)
      }
      return retryResponse.json()
    }
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
  /**
   * Exchange a Supabase JWT for our own short-lived JWT + set HttpOnly refresh cookie.
   * Call this immediately after any Supabase onAuthStateChange event that provides a session.
   */
  exchangeSession: async (supabaseToken: string): Promise<{ access_token: string; token_type: string; user: User }> => {
    const hfToken = process.env.NEXT_PUBLIC_HF_TOKEN
    const headers: Record<string, string> = {
      ...(hfToken && { Authorization: `Bearer ${hfToken}` }),
      ...(!hfToken && { Authorization: `Bearer ${supabaseToken}` }),
      'X-Auth-Token': supabaseToken,
    }
    const res = await fetch(`${API_BASE_URL}/api/auth/session`, {
      method: 'POST',
      credentials: 'include', // ensures the Set-Cookie response header is respected
      headers,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: 'Session exchange failed' }))
      throw new Error(err.message || err.detail || 'Session exchange failed')
    }
    return res.json()
  },

  /**
   * Logout: revoke the refresh token server-side and clear the HttpOnly cookie.
   * Also signs out of Supabase for Google OAuth users.
   */
  logout: async (): Promise<void> => {
    try {
      await fetch(`${API_BASE_URL}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      })
    } catch { /* best-effort */ }
    setApiToken(null)
    supabase.auth.signOut().catch(() => {})
  },

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

  updateProfile: (data: {
    name?: string
    avatar_url?: string
    theme?: string
    current_password?: string
    password?: string
  }) =>
    fetchAPI<{ status: string; user: User }>('/api/auth/profile', {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  /** Upload an avatar image via the backend.
   *  Returns the public URL to save in profiles.avatar_url. */
  uploadAvatar: async (file: File): Promise<{ status: string; avatar_url: string }> => {
    let token = getApiToken()

    const makeHeaders = (t: string | null) => {
      const hfToken = process.env.NEXT_PUBLIC_HF_TOKEN
      return {
        ...(hfToken && { Authorization: `Bearer ${hfToken}` }),
        ...(!hfToken && t && { Authorization: `Bearer ${t}` }),
        ...(t && { 'X-Auth-Token': t }),
      }
    }

    const formData = new FormData()
    formData.append('file', file)

    let response = await fetch(`${API_BASE_URL}/api/auth/upload-avatar`, {
      method: 'POST',
      credentials: 'include',
      headers: makeHeaders(token),
      body: formData,
    })

    // Token-refresh retry
    if (response.status === 401) {
      const freshToken = await refreshToken()
      if (freshToken) {
        token = freshToken
        response = await fetch(`${API_BASE_URL}/api/auth/upload-avatar`, {
          method: 'POST',
          credentials: 'include',
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
      const error = await response.json().catch(() => ({ message: 'Failed to upload avatar' }))
      throw new Error(error.message || error.detail || 'Failed to upload avatar')
    }

    return response.json()
  },
}

export const materialsAPI = {
  list: () => fetchAPI<Material[]>('/api/materials'),

  uploadPDF: async (file: File): Promise<{ material_id: string; title: string; chunks_count: number }> => {
    let token = getApiToken()
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
      credentials: 'include',
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
          credentials: 'include',
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


export const asrAPI = {
  /**
   * Transcribe an audio Blob using the selected language model.
   * @param audioBlob - The recorded audio (webm/wav/ogg from MediaRecorder)
   * @param language  - 'en' for Parakeet, 'ar' for wav2vec2
   */
  transcribe: async (audioBlob: Blob, language: 'en' | 'ar'): Promise<{ transcript: string }> => {
    const token = getApiToken()
    const hfToken = process.env.NEXT_PUBLIC_HF_TOKEN

    const formData = new FormData()
    formData.append('audio', audioBlob, 'recording.webm')
    formData.append('language', language)

    // Build headers without Content-Type (browser sets multipart boundary automatically)
    const headers: Record<string, string> = {}
    if (hfToken) {
      headers['Authorization'] = `Bearer ${hfToken}`
    } else if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
    if (token) headers['X-Auth-Token'] = token

    const response = await fetch(`${API_BASE_URL}/api/asr/transcribe`, {
      method: 'POST',
      credentials: 'include',
      headers,
      body: formData,
    })

    if (response.status === 401) {
      const freshToken = await refreshToken()
      if (freshToken) {
        headers['X-Auth-Token'] = freshToken
        if (!hfToken) headers['Authorization'] = `Bearer ${freshToken}`
        const retry = await fetch(`${API_BASE_URL}/api/asr/transcribe`, {
          method: 'POST',
          credentials: 'include',
          headers,
          body: formData,
        })
        if (!retry.ok) {
          const err = await retry.json().catch(() => ({ message: 'Transcription failed' }))
          throw new Error(err.message || err.detail || 'Transcription failed')
        }
        return retry.json()
      }
      forceSignOut()
      throw new Error('Session expired. Please sign in again.')
    }

    if (!response.ok) {
      const err = await response.json().catch(() => ({ message: 'Transcription failed' }))
      throw new Error(err.message || err.detail || 'Transcription failed')
    }

    return response.json()
  },
}
