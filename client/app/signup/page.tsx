'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Loader2, Sparkles, MessageSquare, Brain, FileText, User, Mail, Lock, Eye, EyeOff } from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from 'sonner'
import { supabase } from '@/lib/supabase'

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

export default function SignUpPage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!name.trim() || !email.trim() || !password) {
      toast.error('Please fill in all required fields.')
      return
    }

    if (password.length < 6) {
      toast.error('Password must be at least 6 characters long.')
      return
    }

    if (password !== confirmPassword) {
      toast.error('Passwords do not match.')
      return
    }

    setIsLoading(true)
    try {
      const redirectTo = typeof window !== 'undefined' ? `${window.location.origin}/auth/callback` : undefined
      const { data, error } = await supabase.auth.signUp({
        email: email.trim(),
        password,
        options: {
          emailRedirectTo: redirectTo,
          data: {
            name: name.trim(),
            full_name: name.trim(),
          },
        },
      })

      if (error) {
        console.error('Supabase SignUp Error:', error)
        const errorMsg =
          typeof error.message === 'string' && error.message.trim() && error.message !== '{}'
            ? error.message
            : (error as any).error_description || JSON.stringify(error)
        toast.error(errorMsg)
      } else if (data.session) {
        toast.success('Account created successfully!')
        router.replace('/dashboard')
      } else {
        toast.success('Registration successful! Please check your email to verify your account.')
        router.replace('/')
      }
    } catch (err: any) {
      const catchMsg = err?.message && typeof err.message === 'string' ? err.message : 'An unexpected error occurred during sign up.'
      toast.error(catchMsg)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex bg-background">
      {/* ── Left panel ── */}
      <div className="hidden lg:flex flex-col justify-between w-[55%] bg-primary/5 border-r border-border/50 px-14 py-12">
        <Logo />

        <div className="space-y-10">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold text-foreground leading-tight mb-4">
              Join Study Buddy today
            </h1>
            <p className="text-muted-foreground text-lg leading-relaxed max-w-md">
              Create an account to upload your materials, chat with PDFs, and generate interactive quizzes.
            </p>
          </motion.div>

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

        <p className="text-xs text-muted-foreground/60">
          Study smarter, not harder.
        </p>
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        <div className="mb-8 w-full max-w-sm lg:hidden">
          <Logo />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full max-w-sm space-y-6"
        >
          {/* Heading */}
          <div className="space-y-1.5 pt-1">
            <h2 className="text-2xl font-bold text-foreground">Create an account</h2>
            <p className="text-muted-foreground text-sm">
              Enter your details to get started with your free account
            </p>
          </div>

          {/* Sign up form */}
          <form onSubmit={handleSignUp} className="space-y-3.5">
            {/* Username */}
            <div className="space-y-1.5">
              <Label htmlFor="name" className="text-xs font-medium text-muted-foreground">
                Username
              </Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="name"
                  type="text"
                  placeholder="Username"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="pl-9 h-10 bg-card/50 border-border/60 focus-visible:ring-primary text-sm"
                  required
                />
              </div>
            </div>

            {/* Email */}
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
                  className="pl-9 h-10 bg-card/50 border-border/60 focus-visible:ring-primary text-sm"
                  required
                />
              </div>
            </div>

            {/* Password */}
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
                  className="pl-9 pr-10 h-10 bg-card/50 border-border/60 focus-visible:ring-primary text-sm"
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

            {/* Confirm Password */}
            <div className="space-y-1.5">
              <Label htmlFor="confirmPassword" className="text-xs font-medium text-muted-foreground">
                Confirm password
              </Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="Confirm password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="pl-9 pr-10 h-10 bg-card/50 border-border/60 focus-visible:ring-primary text-sm"
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

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full h-11 text-sm font-semibold shadow-md transition-all mt-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating account...
                </>
              ) : (
                'Sign Up'
              )}
            </Button>
          </form>

          <p className="text-center text-xs text-muted-foreground pt-1">
            Already have an account?
            <Link href="/" className="ml-1.5 font-semibold text-primary hover:underline">
              Sign in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  )
}
