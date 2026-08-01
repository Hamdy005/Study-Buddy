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

      try {
        const { authAPI } = await import('@/lib/api')
        const result = await authAPI.exchangeSession(session.access_token)
        const profile = result.user

        login(profile, result.access_token)

        const typeParam = searchParams.get('type') || hashParams.get('type')
        const nextParam = searchParams.get('next') || hashParams.get('next')

        if (typeParam === 'recovery' || nextParam === '/update-password' || nextParam?.includes('update-password')) {
          router.replace('/update-password')
        } else {
          router.replace('/dashboard')
        }
      } catch (err) {
        console.error('OAuth callback session exchange failed:', err)
        logout()
        router.replace('/?error=oauth_failed')
      }
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
