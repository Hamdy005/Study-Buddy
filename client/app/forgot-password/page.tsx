'use client'

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Loader2, Mail, ArrowLeft, CheckCircle2 } from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from 'sonner'
import { supabase } from '@/lib/supabase'
import { API_BASE_URL } from '@/lib/api'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email.trim()) {
      toast.error('Please enter your email address.')
      return
    }

    setIsLoading(true)
    try {
      // 1. Enforce in-memory rate limit via backend (3 requests per hour, 60s cooldown)
      const rateLimitRes = await fetch(`${API_BASE_URL}/api/auth/check-email-rate-limit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'forgot_password',
          email: email.trim(),
        }),
      })

      if (!rateLimitRes.ok) {
        const rateLimitData = await rateLimitRes.json().catch(() => ({ detail: 'Rate limit exceeded.' }))
        toast.error(rateLimitData.detail || rateLimitData.message || 'Maximum 3 password reset requests per hour allowed.')
        setIsLoading(false)
        return
      }

      // 2. Trigger Supabase reset email
      const redirectToUrl = typeof window !== 'undefined' ? `${window.location.origin}/auth/callback?next=/update-password` : undefined
      const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
        redirectTo: redirectToUrl,
      })

      if (error) {
        const errorMsg = typeof error.message === 'string' && error.message.trim() && error.message !== '{}' ? error.message : 'Failed to send reset link. Please try again.'
        toast.error(errorMsg)
      } else {
        setIsSubmitted(true)
        toast.success('Password reset link sent to your email!')
      }
    } catch {
      toast.error('Failed to send reset link. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-sm space-y-4"
      >
        <div className="flex justify-center -ml-9">
          <Logo />
        </div>

        <div className="w-full space-y-6 bg-card/40 p-8 rounded-2xl border border-border/50 shadow-xl backdrop-blur-sm">

        {isSubmitted ? (
          <div className="text-center space-y-4 py-2">
            <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <div className="space-y-1.5">
              <h2 className="text-xl font-bold text-foreground">Check your email</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                We sent a password reset link to <span className="font-semibold text-foreground">{email}</span>.
              </p>
            </div>
            <p className="text-xs text-muted-foreground pt-2">
              Didn&apos;t receive the email? Check your spam folder or try again.
            </p>
            <Button
              variant="outline"
              onClick={() => setIsSubmitted(false)}
              className="w-full h-10 text-xs font-medium border-border/60"
            >
              Try another email
            </Button>
          </div>
        ) : (
          <>
            <div className="space-y-1.5">
              <h2 className="text-xl font-bold text-foreground">Forgot password?</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Enter your email address and we&apos;ll send you a link to reset your password.
              </p>
            </div>

            <form onSubmit={handleResetPassword} className="space-y-4">
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

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-11 text-sm font-semibold shadow-md transition-all"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending link...
                  </>
                ) : (
                  'Send Reset Link'
                )}
              </Button>
            </form>
          </>
        )}

        <div className="pt-2 text-center border-t border-border/40">
          <Link
            href="/"
            className="inline-flex items-center text-xs font-medium text-muted-foreground hover:text-primary transition-colors gap-1.5"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to Sign In
          </Link>
        </div>
      </div>
    </motion.div>
  </div>
  )
}
