import React from 'react'
import { cn } from '@/lib/utils'

interface LogoProps {
  className?: string
  iconClassName?: string
  showText?: boolean
  textClassName?: string
}

export function Logo({ className, iconClassName, showText = true, textClassName }: LogoProps) {
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className={cn(
        "w-10 h-10 bg-[#4B6EE5] rounded-lg flex items-center justify-center transition-all duration-300 hover:scale-105 shadow-md",
        iconClassName
      )}>
        <svg 
          viewBox="0 0 32 32" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
          className="w-7 h-7"
        >
          {/* Top of the hat */}
          <ellipse cx="16" cy="12.5" rx="8" ry="2.2" fill="white" transform="rotate(-4, 16, 12.5)"/>
          
          {/* Body of the hat */}
          <path d="M12 12 Q11.5 15.5 12.5 17.5 Q14 19 16 19 Q18 19 19.5 17.5 Q20.5 15.5 20 12 Z" fill="white"/>
          
          {/* Shorter Robe/Tassel Line */}
          <path d="M23.5 12 Q24.5 13 24 14.5 Q23.5 15.5 23 16.5" stroke="white" strokeWidth="1.3" strokeLinecap="round" opacity="0.9"/>
          
        </svg>
      </div>
      {showText && (
        <span className={cn("text-xl font-bold text-foreground tracking-tight", textClassName)}>
          Study Buddy
        </span>
      )}
    </div>
  )
}
