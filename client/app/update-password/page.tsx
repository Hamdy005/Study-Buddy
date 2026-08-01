'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Loader2, Lock, Eye, EyeOff, CheckCircle2 } from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from 'sonner'
import { supabase } from '@/lib/supabase'

export default function UpdatePasswordPage() {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  const router = useRouter()

  useEffect(() => {
    // Listen for PASSWORD_RECOVERY event or check for active session
    const { data: authListener } = supabase.auth.onAuthStateChange((event) => {
      if (event === 'PASSWORD_RECOVERY') {
        toast.info('Please set your new password below.')
      }
    })

    return () => {
      authListener.subscription.unsubscribe()
    }
  }, [])

  const handleUpdatePassword = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!password) {
      toast.error('Please enter a new password.')
      return
    }

    if (password.length < 8) {
      toast.error('Password must be at least 8 characters long.')
      return
    }

    if (password !== confirmPassword) {
      toast.error('Passwords do not match.')
      return
    }

    setIsLoading(true)
    try {
      const { error } = await supabase.auth.updateUser({
        password,
      })

      if (error) {
        toast.error(error.message)
      } else {
        setIsSuccess(true)
        toast.success('Password updated successfully!')
        setTimeout(() => {
          router.replace('/dashboard')
        }, 2000)
      }
    } catch {
      toast.error('Failed to update password. Please try again.')
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
        className="w-full max-w-sm space-y-6 bg-card/40 p-8 rounded-2xl border border-border/50 shadow-xl backdrop-blur-sm"
      >
        <div className="flex justify-center mb-2">
          <Logo className="justify-center" />
        </div>

        {isSuccess ? (
          <div className="text-center space-y-4 py-2">
            <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-6 h-6 text-primary" />
            </div>
            <div className="space-y-1.5">
              <h2 className="text-xl font-bold text-foreground">Password Reset Complete!</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Your password has been updated. Redirecting you to your dashboard...
              </p>
            </div>
            <Loader2 className="w-5 h-5 animate-spin text-primary mx-auto" />
          </div>
        ) : (
          <>
            <div className="space-y-1.5 text-center">
              <h2 className="text-xl font-bold text-foreground">Change Your Password</h2>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Enter a new password below to change your password.
              </p>
            </div>

            <form onSubmit={handleUpdatePassword} className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="password" className="text-xs font-medium text-muted-foreground">
                  New password*
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="New password"
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

              <div className="space-y-1.5">
                <Label htmlFor="confirmPassword" className="text-xs font-medium text-muted-foreground">
                  Re-enter new password*
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Re-enter new password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="pl-9 pr-10 h-11 bg-card/50 border-border/60 focus-visible:ring-primary"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors p-1"
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Password Requirement Box */}
              <div className="p-3.5 bg-card/60 border border-border/70 rounded-xl space-y-2 text-xs">
                <p className="font-medium text-muted-foreground">Your password must contain:</p>
                <div className="flex items-center gap-2">
                  <CheckCircle2
                    className={`w-4 h-4 transition-colors ${
                      password.length >= 8 ? 'text-emerald-500' : 'text-muted-foreground/30'
                    }`}
                  />
                  <span
                    className={`transition-colors ${
                      password.length >= 8
                        ? 'text-emerald-600 dark:text-emerald-400 font-medium'
                        : 'text-muted-foreground'
                    }`}
                  >
                    At least 8 characters in length
                  </span>
                </div>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-11 text-sm font-semibold shadow-md transition-all bg-primary text-primary-foreground hover:bg-primary/90"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Resetting password...
                  </>
                ) : (
                  'Reset password'
                )}
              </Button>
            </form>
          </>
        )}

        {!isSuccess && (
          <div className="pt-2 text-center border-t border-border/40">
            <Link
              href="/"
              className="text-xs font-medium text-muted-foreground hover:text-primary transition-colors"
            >
              Back to Sign In
            </Link>
          </div>
        )}
      </motion.div>
    </div>
  )
}
