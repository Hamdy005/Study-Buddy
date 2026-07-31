'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/contexts/auth-context'
import { Loader2 } from 'lucide-react'

export default function AuthCallbackPage() {
  const router = useRouter()
  const { login, logout } = useAuth()

  useEffect(() => {
    const handleCallback = async () => {
      const searchParams = new URLSearchParams(window.location.search)
      const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'))
      const errorParam =
        searchParams.get('error_description') ||
        hashParams.get('error_description') ||
        searchParams.get('error') ||
        hashParams.get('error')

      if (errorParam) {
        console.error('Auth callback error:', errorParam)
        logout()
        router.replace(`/?error=${encodeURIComponent(errorParam)}`)
        return
      }

      // Supabase exchanges the URL hash/code for a session automatically
      const { data, error } = await supabase.auth.getSession()

      if (error || !data.session) {
        console.error('OAuth callback error:', error)
        logout()
        router.replace('/?error=oauth_failed')
        return
      }

      const session = data.session
      const sbUser = session.user

      // ── Prefer the DB name/avatar (per-user display cache) over Google metadata ────
      // Each account's cache is keyed by user ID so different accounts never mix.
      const userId = sbUser.id
      const DISPLAY_CACHE_KEY = `auth_display_${userId}`
      const PROFILE_CACHE_KEY = 'auth_user'
      const PROFILE_CACHE_TS_KEY = 'auth_user_cached_at'

      let displayName =
        sbUser.user_metadata?.full_name ||
        sbUser.user_metadata?.name ||
        sbUser.email?.split('@')[0] ||
        'User'
      let displayAvatar: string | undefined = sbUser.user_metadata?.avatar_url

      try {
        const raw = localStorage.getItem(DISPLAY_CACHE_KEY)
        if (raw) {
          const display = JSON.parse(raw)
          if (display.name) displayName = display.name
          if (display.avatar !== undefined) displayAvatar = display.avatar
        }
      } catch {}

      const userData = {
        id: sbUser.id,
        name: displayName,
        email: sbUser.email!,
        avatar: displayAvatar,
      }

      // ── Write caches BEFORE calling login() ────────────────────────────────
      // The auth context's onAuthStateChange fires concurrently the moment
      // getSession() resolves. If the profile cache is empty at that point,
      // fetchAndSetProfile() falls through to its optimistic render and briefly
      // shows Google's name/avatar. Pre-populating the cache here ensures the
      // concurrent listener finds data immediately and returns early.
      try {
        localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(userData))
        localStorage.setItem(PROFILE_CACHE_TS_KEY, String(Date.now()))
        localStorage.setItem(DISPLAY_CACHE_KEY, JSON.stringify({ name: displayName, avatar: displayAvatar }))
      } catch {}

      login(userData, session.access_token)

      router.replace('/dashboard')
    }

    handleCallback()
  }, [login, logout, router])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-4 bg-background">
      <Loader2 className="h-10 w-10 animate-spin text-primary" />
      <p className="text-muted-foreground text-sm">Completing sign-in...</p>
    </div>
  )
}
