'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Loader2, Sparkles, MessageSquare, Brain, FileText } from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/contexts/auth-context'

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
    desc: '10 daily AI generations (summary + quiz) with unlimited material chat.',
  },
]

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false)
  const { user, isLoading: isAuthLoading, logout } = useAuth()
  const router = useRouter()

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search)
    const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'))
    if (searchParams.has('error') || hashParams.has('error')) {
      if (user) {
        logout()
      }
      return
    }

    if (!isAuthLoading && user) {
      router.replace('/dashboard')
    }
  }, [user, isAuthLoading, router, logout])

  const handleGoogleSignIn = async () => {
    setIsLoading(true)
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
      })
      if (error) {
        toast.error('Google sign-in failed: ' + error.message)
        setIsLoading(false)
      }
    } catch {
      toast.error('Google sign-in is not available right now')
      setIsLoading(false)
    }
  }

  const hasError = typeof window !== 'undefined' && (
    new URLSearchParams(window.location.search).has('error') ||
    new URLSearchParams(window.location.hash.replace('#', '?')).has('error')
  )

  if (isAuthLoading || (user && !hasError)) {
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
          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-foreground">Welcome back</h2>
            <p className="text-muted-foreground text-sm">
              Sign in to continue your learning journey
            </p>
          </div>

          {hasError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
              Sign In Failed, Try Signing In Again
            </div>
          )}

          {/* Sign in card */}
          <div className="space-y-4">
            <Button
              variant="outline"
              className="w-full h-12 text-sm font-medium border-border/60 hover:bg-muted/60 transition-all"
              onClick={handleGoogleSignIn}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <svg className="mr-2 h-4 w-4 flex-shrink-0" viewBox="0 0 24 24">
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                </svg>
              )}
              Continue with Google
            </Button>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3 pt-4 border-t border-border/40">
            {[
              { value: '10', label: 'Daily generations' },
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
