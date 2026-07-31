'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { Loader2, Sparkles, MessageSquare, Brain, FileText, Mail, Lock, Eye, EyeOff } from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from 'sonner'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/contexts/auth-context'
import Script from 'next/script'
import { useTheme } from 'next-themes'

declare global {
  interface Window {
    google: any
  }
}

const features = [
  {
    icon: FileText,
    title: 'Upload & Understand',
    desc: 'Drop any PDF or paste a URL — get instant structured summaries.',
  },
  {
    icon: MessageSquare,
    title: 'Chat With Your Material',
    desc: 'Ask questions and get answers grounded in your actual content.',
  },
  {
    icon: Brain,
    title: 'Test Your Knowledge',
    desc: 'Auto-generated MCQ and True/False quizzes at any difficulty.',
  },
  {
    icon: Sparkles,
    title: 'Powered by AI',
    desc: '20 daily AI generations (summary + quiz) with unlimited material chat.',
  },
]

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const { user, isLoading: isAuthLoading, logout } = useAuth()
  const router = useRouter()
  const { resolvedTheme } = useTheme()
  const [btnWidth, setBtnWidth] = useState(384)

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email || !password) {
      toast.error('Please enter both email and password')
      return
    }
    setIsSubmitting(true)
    setErrorMessage(null)
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      })
      if (error) {
        toast.error(error.message)
      } else {
        toast.success('Signed in successfully!')
        router.replace('/dashboard')
      }
    } catch {
      toast.error('Sign in failed. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search)
    const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'))
    const rawError =
      searchParams.get('error_description') ||
      hashParams.get('error_description') ||
      searchParams.get('error') ||
      hashParams.get('error')

    if (rawError) {
      if (user) {
        logout()
      }
      const decoded = decodeURIComponent(rawError.replace(/\+/g, ' '))
      if (decoded === 'oauth_failed') {
        setErrorMessage('Google sign-in failed. Please try signing in again.')
      } else {
        setErrorMessage(decoded)
      }
      try {
        window.history.replaceState({}, document.title, window.location.pathname)
      } catch {}
      return
    }

    if (!isAuthLoading && user) {
      router.replace('/dashboard')
    }
  }, [user, isAuthLoading, router, logout])

  const handleCredentialResponse = useCallback(async (response: any) => {
    setIsLoading(true)
    setErrorMessage(null)
    try {
      const { error } = await supabase.auth.signInWithIdToken({
        provider: 'google',
        token: response.credential,
      })
      if (error) {
        toast.error('Google sign-in failed: ' + error.message)
        setIsLoading(false)
      }
    } catch {
      toast.error('Google sign-in is not available right now')
      setIsLoading(false)
    }
  }, [])

  const initGoogleSignIn = useCallback(() => {
    if (typeof window !== 'undefined' && window.google) {
      window.google.accounts.id.initialize({
        client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
        callback: handleCredentialResponse,
      })

      const btnContainer = document.getElementById('google-signin-btn')
      if (btnContainer) {
        window.google.accounts.id.renderButton(
          btnContainer,
          {
            theme: resolvedTheme === 'dark' ? 'filled_black' : 'outline',
            size: 'large',
            width: btnWidth,
            text: 'continue_with',
            shape: 'pill',
          }
        )
      }
    }
  }, [resolvedTheme, btnWidth, handleCredentialResponse])

  useEffect(() => {
    const updateWidth = () => {
      const container = document.getElementById('google-signin-container')
      if (container) {
        const width = Math.min(400, Math.max(200, container.clientWidth || 384))
        setBtnWidth(width)
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  useEffect(() => {
    if (isAuthLoading || isLoading) return
    const timer = setTimeout(initGoogleSignIn, 100)
    return () => clearTimeout(timer)
  }, [isAuthLoading, isLoading, initGoogleSignIn])

  if (isAuthLoading || (user && !errorMessage)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex bg-background">

      {/* ── Left panel ── */}
      <div className="hidden lg:flex flex-col justify-between w-[55%] bg-primary/5 border-r border-border/50 px-14 py-12">
        {/* Logo */}
        <Logo />

        {/* Hero text */}
        <div className="space-y-10">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold text-foreground leading-tight mb-4">
              Your AI-powered<br />study companion
            </h1>
            <p className="text-muted-foreground text-lg leading-relaxed max-w-md">
              Upload your materials, chat with them, generate quizzes, and get summaries — all in one place.
            </p>
          </motion.div>

          {/* Feature list */}
          <div className="grid grid-cols-1 gap-4">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.15 + i * 0.1 }}
                className="flex items-start gap-4 p-4 rounded-xl bg-card/60 border border-border/40"
              >
                <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <f.icon className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <p className="font-semibold text-foreground text-sm">{f.title}</p>
                  <p className="text-muted-foreground text-sm mt-0.5 leading-relaxed">{f.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Bottom quote */}
        <p className="text-xs text-muted-foreground/60">
          Study smarter, not harder.
        </p>
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        {/* Mobile logo */}
        <div className="mb-10 w-full max-w-sm lg:hidden">
          <Logo />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full max-w-sm space-y-8"
        >

          {/* Heading */}
          <div className="space-y-1.5 pt-1">
            <h2 className="text-2xl font-bold text-foreground">Welcome back</h2>
            <p className="text-muted-foreground text-sm">
              Sign in to continue your learning journey
            </p>
          </div>

          {errorMessage && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
              {errorMessage}
            </div>
          )}

          {/* Sign in form & OAuth */}
          <div className="space-y-5">
            <form onSubmit={handleEmailSignIn} className="space-y-3.5">
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-xs font-medium text-muted-foreground">
                  Email address
                </Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-9 h-11 bg-card/50 border-border/60 focus-visible:ring-primary"
                    required
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="password" className="text-xs font-medium text-muted-foreground">
                  Password
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-9 pr-10 h-11 bg-card/50 border-border/60 focus-visible:ring-primary"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors p-1"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <Button
                type="submit"
                disabled={isSubmitting || isLoading}
                className="w-full h-11 text-sm font-semibold shadow-md transition-all"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  'Sign In'
                )}
              </Button>

              <div className="flex items-center justify-between text-xs pt-1 gap-4">
                <span className="text-muted-foreground shrink-0">
                  Don&apos;t have an account?
                  <Link href="/signup" className="ml-1.5 font-semibold text-primary hover:underline">
                    Sign up
                  </Link>
                </span>
                <Link
                  href="/forgot-password"
                  className="font-medium text-primary hover:underline transition-all shrink-0"
                >
                  Forgot password?
                </Link>
              </div>
            </form>

            <div className="relative flex items-center justify-center py-1">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-border/40" />
              </div>
              <div className="relative bg-background px-3 text-[11px] font-medium text-muted-foreground/80 uppercase tracking-wider">
                Or continue with
              </div>
            </div>

            <Script
              src="https://accounts.google.com/gsi/client"
              strategy="afterInteractive"
              onLoad={initGoogleSignIn}
            />
            {isLoading ? (
              <Button
                variant="outline"
                className="w-full h-11 text-sm font-medium border-border/60 hover:bg-muted/60 transition-all flex items-center justify-center"
                disabled
              >
                <Loader2 className="mr-2 h-4 w-4 animate-spin text-primary" />
                Signing you in...
              </Button>
            ) : (
              <div className="relative w-full h-11">
                {/* Premium Custom Static Button */}
                <Button
                  variant="outline"
                  className="w-full h-11 text-sm font-medium border-border/60 hover:bg-muted/60 transition-all flex items-center justify-center gap-3"
                >
                  <svg className="h-5 w-5" viewBox="0 0 24 24">
                    <path
                      fill="#4285F4"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="#34A853"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="#FBBC05"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                    />
                    <path
                      fill="#EA4335"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                    />
                  </svg>
                  Continue with Google
                </Button>

                {/* Invisible Google Button Overlay to handle clicks securely */}
                <div
                  id="google-signin-container"
                  className="absolute inset-0 w-full h-full opacity-0 z-10 cursor-pointer"
                  style={{ colorScheme: 'light' }}
                >
                  <div
                    id="google-signin-btn"
                    className="w-full h-full cursor-pointer flex justify-center"
                    style={{ colorScheme: 'light' }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3 pt-4  mt-6 border-t border-border/40">
            {[
              { value: '20', label: 'Daily generations' },
              { value: 'Free', label: 'to get started' },
              { value: '∞', label: 'Chat messages' },
            ].map((stat) => (
              <div key={stat.label} className="text-center space-y-1">
                <p className={`font-bold text-primary ${stat.value === '∞' ? 'text-xl leading-none -mb-0.5' : 'text-sm'}`}>
                  {stat.value}
                </p>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
