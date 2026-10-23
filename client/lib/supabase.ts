import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Resolves a clean, verified origin for auth callbacks and transactional email redirection.
export function getAuthRedirectUrl(path: string = '/auth/callback'): string {
  const cleanPath = path.startsWith('/') ? path : `/${path}`
  const envSiteUrl = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, '')

  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname

    // Local development loopback (stays local)
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return `${window.location.origin}${cleanPath}`
    }

    // If accessed through Hugging Face Spaces or other backend proxy URLs,
    // force redirection to the official verified frontend custom domain.
    if (hostname.includes('hf.space') || hostname.includes('huggingface.co')) {
      return `https://studybuddyai.dev${cleanPath}`
    }

    // If an explicit site URL is configured, use it for production
    if (envSiteUrl) {
      return `${envSiteUrl}${cleanPath}`
    }

    return `${window.location.origin}${cleanPath}`
  }

  // Fallback during server-side evaluation
  return envSiteUrl ? `${envSiteUrl}${cleanPath}` : `https://studybuddyai.dev${cleanPath}`
}
