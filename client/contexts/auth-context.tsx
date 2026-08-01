'use client'

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import { useTheme } from 'next-themes'
import { supabase } from '@/lib/supabase'
import { authAPI, setApiToken, onForceSignOut } from '@/lib/api'

export interface UserData {
  id: string
  name: string
  email: string
  avatar?: string
  theme?: 'light' | 'dark' | 'system'
  usage?: {
    used: number
    limit: number
    remaining: number
  }
}

interface AuthContextType {
  user: UserData | null
  token: string | null
  login: (user: UserData, token: string) => void
  logout: () => Promise<void>
  updateUser: (data: Partial<UserData>) => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType | null>(null)

// ── Display-only cache (name + avatar), keyed per user ─────────────────────
// Keyed as `auth_display_<userId>` so different accounts never bleed.
// Contains NO sensitive data — purely for instant UI render on next visit.
// Intentionally persisted through logout.

function displayCacheKey(userId: string): string {
  return `auth_display_${userId}`
}

function getDisplayCache(userId: string): { name: string; avatar?: string } | null {
  try {
    const raw = localStorage.getItem(displayCacheKey(userId))
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function setDisplayCache(userId: string, name: string, avatar?: string) {
  try {
    localStorage.setItem(displayCacheKey(userId), JSON.stringify({ name, avatar }))
  } catch {}
}

// ── Non-sensitive profile cache (no tokens) ────────────────────────────────
const PROFILE_CACHE_KEY    = 'auth_user'
const PROFILE_CACHE_TS_KEY = 'auth_user_cached_at'
const PROFILE_CACHE_TTL_MS = 60_000 // 60 seconds

function getCachedProfile(): UserData | null {
  try {
    const ts = Number(localStorage.getItem(PROFILE_CACHE_TS_KEY) ?? 0)
    if (Date.now() - ts > PROFILE_CACHE_TTL_MS) return null
    const raw = localStorage.getItem(PROFILE_CACHE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as UserData
  } catch {
    return null
  }
}

function setCachedProfile(user: UserData) {
  localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(user))
  localStorage.setItem(PROFILE_CACHE_TS_KEY, String(Date.now()))
  setDisplayCache(user.id, user.name, user.avatar)
}

function bustProfileCache() {
  localStorage.removeItem(PROFILE_CACHE_TS_KEY)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  // ── The access JWT lives ONLY here — never in localStorage ────────────────
  const [token, setToken] = useState<string | null>(null)
  const [user, setUser]   = useState<UserData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { setTheme } = useTheme()

  useEffect(() => {
    if (user?.theme) setTheme(user.theme)
  }, [user?.theme, setTheme])

  // ── Helper: keep module-level token store in sync ─────────────────────────
  const applyToken = useCallback((t: string | null) => {
    setToken(t)
    setApiToken(t) // also updates the api.ts module-level variable
  }, [])

  // ── Fetch the full backend profile and populate state ─────────────────────
  const fetchAndApplyProfile = useCallback(async (isActive: () => boolean) => {
    const cached = getCachedProfile()
    if (cached && isActive()) {
      setUser(cached)
      return
    }
    try {
      const res = await authAPI.getProfile()
      if (res.user && isActive()) {
        if (!(res.user as any)._is_fallback) {
          setCachedProfile(res.user)
        }
        setUser(res.user)
      }
    } catch {
      // Profile fetch failed — user state from optimistic render is still visible
    }
  }, [])

  // ── Listen for force sign-out events from API interceptors ─────────────────
  useEffect(() => {
    const unsubscribe = onForceSignOut(() => {
      setUser(null)
      applyToken(null)
      bustProfileCache()
      localStorage.removeItem(PROFILE_CACHE_KEY)
      localStorage.removeItem('cached_materials')
    })
    return unsubscribe
  }, [applyToken])

  // ── On mount: try silent re-auth via the HttpOnly refresh cookie ───────────
  // This is the key behaviour: on a hard page refresh the JWT is gone from
  // memory, but the HttpOnly cookie survives. We ask our backend to issue a
  // new JWT without any user interaction.
  useEffect(() => {
    let active = true

    const searchParams = new URLSearchParams(window.location.search)
    const hashParams   = new URLSearchParams(window.location.hash.replace('#', '?'))
    if (searchParams.has('error') || hashParams.has('error')) {
      setUser(null)
      applyToken(null)
      setIsLoading(false)
      return
    }

    const trySilentRefresh = async () => {
      try {
        const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || '').replace(/\/$/, '')
        const res = await fetch(`${API_BASE}/api/auth/refresh`, {
          method: 'POST',
          credentials: 'include', // sends the HttpOnly cookie automatically
        })
        if (!res.ok) {
          if (active) {
            setUser(null)
            applyToken(null)
            bustProfileCache()
            localStorage.removeItem(PROFILE_CACHE_KEY)
            localStorage.removeItem('cached_materials')
            setIsLoading(false)
          }
          return
        }
        const data = await res.json()
        const newToken: string = data.access_token
        if (!newToken || !active) {
          if (active) {
            setUser(null)
            applyToken(null)
            setIsLoading(false)
          }
          return
        }
        applyToken(newToken)

        // Show name/avatar from display cache instantly while profile loads
        const cachedProfile = getCachedProfile()
        if (cachedProfile && active) {
          setUser(cachedProfile)
        }

        await fetchAndApplyProfile(() => active)
      } catch {
        // No cookie / network error — treat as logged out
        if (active) {
          setUser(null)
          applyToken(null)
          bustProfileCache()
          localStorage.removeItem(PROFILE_CACHE_KEY)
          localStorage.removeItem('cached_materials')
        }
      } finally {
        if (active) setIsLoading(false)
      }
    }

    trySilentRefresh()
    return () => { active = false }
  }, [applyToken, fetchAndApplyProfile])

  // ── Supabase onAuthStateChange: exchange Supabase token for our JWT ────────
  // This fires after any Supabase sign-in (Google OAuth, email magic-link, etc.)
  // We exchange the Supabase token once for our own short-lived JWT + HttpOnly
  // refresh cookie and from that point on use our tokens only.
  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search)
    const hashParams   = new URLSearchParams(window.location.hash.replace('#', '?'))
    if (searchParams.has('error') || hashParams.has('error')) {
      supabase.auth.signOut().catch(() => {})
      return
    }

    let isActive = true

    const { data: listener } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (event === 'SIGNED_OUT' || !session) {
        if (!isActive) return
        setUser(null)
        applyToken(null)
        bustProfileCache()
        localStorage.removeItem(PROFILE_CACHE_KEY)
        localStorage.removeItem('cached_materials')
        return
      }

      // We only need to exchange the token once per new Supabase session.
      // If we already have our own JWT in state, the session is already exchanged.
      if (token) return

      try {
        const result = await authAPI.exchangeSession(session.access_token)
        if (!isActive) return

        applyToken(result.access_token)

        // Use profile returned by the exchange endpoint directly (saves an extra round-trip)
        const profile = result.user as UserData
        if (profile) {
          setUser(profile)
          setCachedProfile(profile)
        } else {
          await fetchAndApplyProfile(() => isActive)
        }

        // Optimistic display cache for instant render on next visit
        if (profile?.id) {
          const display = getDisplayCache(profile.id)
          if (!display) {
            setDisplayCache(profile.id, profile.name, profile.avatar)
          }
        }
      } catch (err) {
        console.error('Session exchange failed:', err)
        if (!isActive) return
        setUser(null)
        applyToken(null)
        localStorage.removeItem(PROFILE_CACHE_KEY)
      }
    })

    return () => {
      isActive = false
      listener?.subscription.unsubscribe()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // intentionally empty — this registers the listener once on mount

  // ── Public API ─────────────────────────────────────────────────────────────

  const login = useCallback((userData: UserData, authToken: string) => {
    setUser(userData)
    applyToken(authToken)
    setCachedProfile(userData)
  }, [applyToken])

  const logout = useCallback(async () => {
    // 1. Revoke refresh token in DB and clear HttpOnly cookie
    await authAPI.logout()
    // 2. Clear local state
    setUser(null)
    applyToken(null)
    bustProfileCache()
    localStorage.removeItem(PROFILE_CACHE_KEY)
    localStorage.removeItem('cached_materials')
    // NOTE: per-user display caches are intentionally kept — no sensitive data,
    // each is keyed by user ID and provides instant name/avatar on next login.
  }, [applyToken])

  const updateUser = useCallback((data: Partial<UserData>) => {
    setUser(prev => {
      if (!prev) return prev
      const updated = { ...prev, ...data }
      setCachedProfile(updated)
      bustProfileCache()
      return updated
    })
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, login, logout, updateUser, isLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
