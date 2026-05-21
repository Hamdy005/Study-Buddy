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
      const errorParam = searchParams.get('error') || hashParams.get('error')

      if (errorParam) {
        console.error('OAuth callback error:', errorParam)
        logout()
        router.replace(`/?error=${errorParam}`)
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

      // ── Prefer the DB name/avatar (display cache) over Google metadata ──────
      const DISPLAY_CACHE_KEY = 'auth_display'
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

      login(
        {
          id: sbUser.id,
          name: displayName,
          email: sbUser.email!,
          avatar: displayAvatar,
        },
        session.access_token
      )

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
