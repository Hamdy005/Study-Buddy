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

      try {
        const res = await authAPI.getProfile()
        if (res.user && isActive) {
          // Don't persist a fallback profile — wait for the real one
          if ((res.user as any)._is_fallback) {
            setUser(res.user)
            return
          }
          setUser(res.user)
          setCachedProfile(res.user)
        }
      } catch {
        // Profile fetch failed — show Google metadata temporarily (not persisted to cache)
        const fallback: UserData = {
          id: sbUser.id,
          name:
            sbUser.user_metadata?.full_name ||
            sbUser.user_metadata?.name ||
            sbUser.email?.split('@')[0] ||
            'User',
          email: sbUser.email || '',
          avatar: sbUser.user_metadata?.avatar_url,
        }
        if (isActive) {
          setUser(fallback)
        }
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
    setCachedProfile(userData)
  }

  const logout = () => {
    supabase.auth.signOut().catch(() => {})
    setUser(null)
    setToken(null)
    localStorage.removeItem('auth_token')
    localStorage.removeItem(PROFILE_CACHE_KEY)
    localStorage.removeItem('cached_materials')
    bustProfileCache()
  }

  const updateUser = (data: Partial<UserData>) => {
    if (!user) return
    const updated = { ...user, ...data }
    setUser(updated)
    setCachedProfile(updated)
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
