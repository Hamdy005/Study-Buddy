'use client'

import { useState, useEffect } from 'react'
import { Zap } from 'lucide-react'
import { usageAPI } from '@/lib/api'
import { cn } from '@/lib/utils'
import { useAuth } from '@/contexts/auth-context'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

export function UsageStats() {
  const { user, updateUser } = useAuth()
  const [usage, setUsage] = useState<{ used: number; limit: number; remaining: number } | null>(() => {
    // 1. Prefer data from user context (most fresh and pre-fetched)
    if (user?.usage) return user.usage

    // 2. Fallback to localStorage cache
    if (typeof window !== 'undefined') {
      const cached = localStorage.getItem('usage_cache')
      if (cached) {
        try {
          return JSON.parse(cached)
        } catch {
          return null
        }
      }
    }
    return null
  })
  const [isLoading, setIsLoading] = useState(!usage)

  useEffect(() => {
    if (user?.usage) {
      setUsage(user.usage)
    }
  }, [user?.usage])

  useEffect(() => {
    // If we have usage from user context, we might still want to refresh it occasionally
    // but we can skip the very first loading state.
    const fetchUsage = async () => {
      try {
        const data = await usageAPI.getUsage()
        setUsage(data)
        updateUser({ usage: data })
        localStorage.setItem('usage_cache', JSON.stringify(data))
      } catch (err) {
        console.error('Failed to fetch usage:', err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchUsage()
    
    // Refresh every minute to keep it somewhat updated
    const interval = setInterval(fetchUsage, 60000)
    return () => clearInterval(interval)
  }, [updateUser])

  if (isLoading || !usage) return null

  // If used by admin (no limit), we might want to hide it or show "Unlimited"
  // But for now we just show the count.
  
  const percentage = (usage.used / usage.limit) * 100
  const isNearLimit = usage.remaining <= 2
  const isAtLimit = usage.remaining === 0

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-card border border-border rounded-full shadow-sm hover:border-primary/30 transition-all cursor-default">
            <Zap className={cn(
              "w-3.5 h-3.5",
              isAtLimit ? "text-destructive" : isNearLimit ? "text-yellow-500" : "text-primary"
            )} />
            <div className="flex flex-col gap-0.5">
              <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-wider leading-none">
                AI Generation
              </span>
              <span className={cn(
                "text-[13px] font-black leading-none",
                isAtLimit ? "text-destructive" : "text-foreground"
              )}>
                {usage.used} / {usage.limit}
              </span>
            </div>
            
            <div className="w-12 h-1 bg-muted rounded-full overflow-hidden ml-1 hidden xs:block">
              <div 
                className={cn(
                  "h-full transition-all duration-500",
                  isAtLimit ? "bg-destructive" : isNearLimit ? "bg-yellow-500" : "bg-primary"
                )}
                style={{ width: `${Math.min(100, percentage)}%` }}
              />
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="p-4 max-w-[260px] bg-white dark:bg-zinc-950 border-2 shadow-xl rounded-xl">
          <div className="space-y-2">
            <p className="font-bold text-base text-foreground">Daily Generation Limit</p>
            <p className="text-sm text-muted-foreground leading-relaxed">
              You have used {usage.used} out of your {usage.limit} daily <span className="font-bold text-foreground">summaries and quizzes</span>. 
            </p>
            <p className="text-xs text-primary font-bold">
              Note: AI Chat is unlimited!
            </p>
            <div className="pt-2 flex items-center justify-between text-xs uppercase font-black text-muted-foreground border-t border-border mt-1">
              <span>Remaining</span>
              <span className={cn(isNearLimit ? "text-yellow-500" : "text-primary")}>
                {usage.remaining}
              </span>
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
