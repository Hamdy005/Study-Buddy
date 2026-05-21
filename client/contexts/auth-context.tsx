'use client'

import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'
import { useTheme } from 'next-themes'
import { supabase } from '@/lib/supabase'
import { authAPI } from '@/lib/api'

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
  logout: () => void
  updateUser: (data: Partial<UserData>) => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType | null>(null)

const PROFILE_CACHE_KEY = 'auth_user'
const PROFILE_CACHE_TS_KEY = 'auth_user_cached_at'
const PROFILE_CACHE_TTL_MS = 60_000 // 60 seconds

// ── Display-only cache (name + avatar) ────────────────────────────────────────
// This cache is intentionally kept through logout — it contains NO sensitive data.
// Its sole purpose is to show the user's name/avatar INSTANTLY on every page load
// without waiting for any async session or API calls to complete.
const DISPLAY_CACHE_KEY = 'auth_display'

function getDisplayCache(): { name: string; avatar?: string } | null {
  try {
    const raw = localStorage.getItem(DISPLAY_CACHE_KEY)
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function setDisplayCache(name: string, avatar?: string) {
  try {
    localStorage.setItem(DISPLAY_CACHE_KEY, JSON.stringify({ name, avatar }))
  } catch {}
}

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
  // Always keep the display cache in sync with the latest real profile
  setDisplayCache(user.name, user.avatar)
}

function bustProfileCache() {
  localStorage.removeItem(PROFILE_CACHE_TS_KEY)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserData | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { setTheme } = useTheme()

  useEffect(() => {
    if (user?.theme) {
      setTheme(user.theme)
    }
  }, [user?.theme, setTheme])

  // ── Hydrate from localStorage on first render ──────────────────────────────
  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search)
    const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'))
    if (searchParams.has('error') || hashParams.has('error')) {
      localStorage.removeItem('auth_token')
      localStorage.removeItem(PROFILE_CACHE_KEY)
      bustProfileCache()
      setIsLoading(false)
      return
    }

    const storedToken = localStorage.getItem('auth_token')
    const storedUser = localStorage.getItem(PROFILE_CACHE_KEY)
    if (storedToken && storedUser) {
      try {
        setToken(storedToken)
        setUser(JSON.parse(storedUser))
      } catch {
        localStorage.removeItem('auth_token')
        localStorage.removeItem(PROFILE_CACHE_KEY)
        bustProfileCache()
      }
    } else if (storedToken) {
      // Token exists but no full cached profile (e.g. right after sign-in or cache expired).
      // Immediately show the display cache (name + avatar) so the UI isn't blank
      // while the async Supabase session + API calls complete.
      const display = getDisplayCache()
      if (display) {
        // Partial user — id/email will be filled in once the real profile arrives.
        // We use an empty string for id so the rest of the UI doesn't break.
        setUser({ id: '', name: display.name, email: '', avatar: display.avatar })
      }
    }
    setIsLoading(false)
  }, [])

  // ── Supabase session sync ──────────────────────────────────────────────────
  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search)
    const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'))
    if (searchParams.has('error') || hashParams.has('error')) {
      supabase.auth.signOut().catch(() => {})
      return
    }

    let isActive = true

    const fetchAndSetProfile = async (authToken: string, sbUser: any) => {
      // Check the short-lived cache first to skip redundant API calls
      const cached = getCachedProfile()
      if (cached) {
        if (isActive) {
          setUser(cached)
        }
        return
      }

      // ── Optimistic render ───────────────────────────────────────────────────
      // Prefer the display cache (user's custom DB name/avatar) over OAuth  metadata (Google name/photo). 
      //  Only fall back to Google data on the very first sign-in when no display cache exists yet.
      if (isActive && sbUser) {
        const display = getDisplayCache()
        const optimistic: UserData = {
          id: sbUser.id,
          name:
            display?.name ||
            sbUser.user_metadata?.full_name ||
            sbUser.user_metadata?.name ||
            sbUser.email?.split('@')[0] ||
            'User',
          email: sbUser.email || '',
          avatar:
            display?.avatar ??
            sbUser.user_metadata?.avatar_url,
        }
        setUser(optimistic)
        // Only seed the display cache on the very first sign-in (no existing cache).
        // Never overwrite it with OAuth data — the real DB profile will update it.
        if (!display) {
          setDisplayCache(optimistic.name, optimistic.avatar)
        }
      }

      try {
        const res = await authAPI.getProfile()
        if (res.user && isActive) {
          // Don't persist a fallback profile — wait for the real one
          if ((res.user as any)._is_fallback) {
            setUser(res.user)
            return
          }
          setUser(res.user)
          setCachedProfile(res.user) // also updates display cache
        }
      } catch {
        // Profile fetch failed — the optimistic value set above is already visible.
      }
    }

    const syncSupabaseSession = async () => {
      try {
        const { data } = await supabase.auth.getSession()
        if (!isActive) return
        const session = data.session
        if (!session) return
        const authToken = session.access_token
        setToken(authToken)
        localStorage.setItem('auth_token', authToken)
        await fetchAndSetProfile(authToken, session.user)
      } catch {
        // supabase not available — nothing to sync
      }
    }

    syncSupabaseSession()

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) return
      const authToken = session.access_token
      setToken(authToken)
      localStorage.setItem('auth_token', authToken)
      fetchAndSetProfile(authToken, session.user)
    })

    return () => {
      isActive = false
      listener?.subscription.unsubscribe()
    }
  }, [])

  const login = (userData: UserData, authToken: string) => {
    setUser(userData)
    setToken(authToken)
    localStorage.setItem('auth_token', authToken)
    setCachedProfile(userData) // also updates display cache
  }

  const logout = () => {
    supabase.auth.signOut().catch(() => {})
    setUser(null)
    setToken(null)
    localStorage.removeItem('auth_token')
    localStorage.removeItem(PROFILE_CACHE_KEY)
    localStorage.removeItem('cached_materials')
    bustProfileCache()
    // NOTE: We intentionally do NOT remove DISPLAY_CACHE_KEY here.
    // The display cache (name + avatar only) is non-sensitive and keeping it
    // means the user's name/avatar appears instantly on their next sign-in.
  }

  const updateUser = (data: Partial<UserData>) => {
    if (!user) return
    const updated = { ...user, ...data }
    setUser(updated)
    setCachedProfile(updated) // also updates display cache
    // Bust TTL so the next auth event re-fetches fresh data from the DB
    bustProfileCache()
  }

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
