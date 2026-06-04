'use client'

import DOMPurify from 'dompurify'
import { useState, useRef, useEffect, use } from 'react'
import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  FileText,
  Link as LinkIcon,
  Sparkles,
  MessageSquare,
  Brain,
  Loader2,
  Send,
  CheckCircle2,
  XCircle,
  ChevronRight,
  ChevronDown,
  SquarePen,
  Printer,
  MessageSquarePlus,
  Pencil,
  Trash2,
  Plus,
  Minus,
  PanelLeftClose,
  PanelLeft,
  MoreHorizontal,
} from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { toast } from 'sonner'
import { Separator } from '@/components/ui/separator'
import { UserDropdown } from '@/components/user-dropdown'
import { UsageStats } from '@/components/usage-stats'
import { useAuth } from '@/contexts/auth-context'
import { materialsAPI, tutorAPI, quizAPI, usageAPI } from '@/lib/api'
import type { Material, ChatMessage, QuizQuestion, QuizResult, ChatSession } from '@/lib/api'

function renderMarkdown(text: string) {
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  let key = 0
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    const headerMatch = trimmed.match(/^(#{1,3})\s*(.+)/)
    if (headerMatch) {
      const level = headerMatch[1].length
      const content = headerMatch[2].trim()
      if (level === 1) {
        elements.push(<h1 key={key++} className="font-bold text-xl mt-4 mb-2">{content}</h1>)
      } else if (level === 2) {
        elements.push(<h2 key={key++} className="font-bold text-lg mt-4 mb-2">{content}</h2>)
      } else {
        elements.push(<h3 key={key++} className="font-bold text-base mt-3 mb-1">{content}</h3>)
      }
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('+ ')) {
      const content = DOMPurify.sanitize(trimmed.slice(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'))
      elements.push(<li key={key++} className="ml-4 list-disc leading-relaxed" dangerouslySetInnerHTML={{ __html: content }} />)
    } else if (trimmed.match(/^\d+\.\s/)) {
      // Remove the number AND any accidental bullets that follow it (ignore ** so we don't break bolding)
      const content = DOMPurify.sanitize(trimmed.replace(/^\d+\.\s*(?:[•\-\–\—\+]\s*|\*(?!\*)\s*)?/, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'))
      elements.push(<li key={key++} className="ml-4 list-decimal leading-relaxed" dangerouslySetInnerHTML={{ __html: content }} />)
    } else if (trimmed === '') {
      elements.push(<div key={key++} className="h-2" />)
    } else {
      const content = DOMPurify.sanitize(line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'))
      elements.push(<p key={key++} dir="auto" className="leading-relaxed" dangerouslySetInnerHTML={{ __html: content }} />)
    }
  }
  return elements
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
}

function printAsPDF(title: string, contentHtml: string) {
  const safeTitle = escapeHtml(title)
  const htmlContent = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>${safeTitle}</title>
      <style>
        body { font-family: system-ui, -apple-system, sans-serif; padding: 60px 40px; color: #1a1a2e; line-height: 1.6; background: #ffffff; }
        h1 { font-size: 24px; margin-bottom: 8px; color: #0f172a; }
        .meta { color: #64748b; font-size: 14px; margin-bottom: 24px; }
        hr { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }
        
        /* Chat Styles */
        .chat-msg { margin-bottom: 16px; padding: 12px 16px; border-radius: 12px; max-width: 90%; }
        .chat-user { background: #4f46e5; color: #ffffff; margin-left: auto; }
        .chat-ai { background: #f1f5f9; color: #1a1a2e; margin-right: auto; }
        .chat-role { font-weight: 700; font-size: 12px; margin-bottom: 4px; display: block; text-transform: uppercase; letter-spacing: 0.05em; }
        
        /* Quiz Styles */
        .stats { display: flex; gap: 16px; margin-bottom: 32px; }
        .stat { flex: 1; text-align: center; padding: 16px; border-radius: 12px; background: #f8fafc; border: 1px solid #e2e8f0; }
        .stat-value { font-size: 24px; font-weight: 700; color: #0f172a; }
        .stat-label { font-size: 12px; color: #64748b; margin-top: 4px; }
        .stat.success { background: #f0fdf4; border-color: #bbf7d0; }
        .stat.success .stat-value { color: #16a34a; }
        .stat.error { background: #fef2f2; border-color: #fecaca; }
        .stat.error .stat-value { color: #dc2626; }
        
        .question-card { margin-bottom: 20px; padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0; }
        .question-card.correct { background: #f0fdf4; border-color: #bbf7d0; }
        .question-card.incorrect { background: #fef2f2; border-color: #fecaca; }
        .question-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
        .q-badge { font-size: 10px; padding: 2px 8px; border-radius: 99px; background: #e2e8f0; color: #475569; font-weight: 600; }
        .q-text { font-weight: 600; color: #0f172a; margin: 0; }
        .option { padding: 4px 0; padding-left: 20px; color: #334155; }
        .result-text { margin-top: 12px; font-size: 14px; font-weight: 600; }
        .result-text.correct { color: #16a34a; }
        .result-text.incorrect { color: #dc2626; }
        
        /* Summary Styles */
        .summary-h2 { font-size: 20px; color: #4f46e5; margin-top: 32px; margin-bottom: 12px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.025em; }
        .summary-h4 { font-size: 16px; color: #1e293b; margin-top: 16px; margin-bottom: 8px; font-weight: 700; }
        .summary-h4 span { color: #4f46e5; margin-right: 8px; }
        .summary-p { margin-bottom: 12px; color: #334155; font-size: 15px; }
        .summary-list-item { margin-left: 0; margin-bottom: 10px; color: #334155; padding-left: 20px; position: relative; font-size: 15px; }
        .summary-list-item::before { content: "•"; position: absolute; left: 0; color: #1a1a2e; font-weight: bold; }
        .summary-numbered-item { margin-left: 0; margin-bottom: 10px; color: #334155; font-size: 15px; }
        .topic-bold { font-weight: 700; color: #0f172a; }
        
        @media print { 
          @page { margin: 0; }
          body { padding: 0; margin: 0; }
          table.print-container { width: 100%; border: none; border-collapse: collapse; }
          thead.print-header { display: table-header-group; }
          tfoot.print-footer { display: table-footer-group; }
          .print-header-space { height: 2cm; }
          .print-footer-space { height: 2cm; }
          .stat, .question-card, .chat-msg { 
            -webkit-print-color-adjust: exact; 
            print-color-adjust: exact; 
            page-break-inside: avoid;
            break-inside: avoid;
          }
        }
      </style>
    </head>
    <body>
      <table class="print-container">
        <thead class="print-header"><tr><td><div class="print-header-space"></div></td></tr></thead>
        <tbody>
          <tr><td style="padding: 0 20mm;">
            <h1>${safeTitle}</h1>
            <div class="meta" style="margin-top: 10px;">Exported from Study Buddy on ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</div>
            ${contentHtml}
            <hr>
            <div class="meta" style="text-align: center;">Generated by Study Buddy - AI-Powered Learning</div>
          </td></tr>
        </tbody>
        <tfoot class="print-footer"><tr><td><div class="print-footer-space"></div></td></tr></tfoot>
      </table>
      <script>
        window.onload = () => {
          setTimeout(() => {
            window.print();
          }, 500);
        };
      </script>
    </body>
    </html>
  `;

  const iframe = document.createElement('iframe');
  iframe.style.position = 'fixed';
  iframe.style.right = '0';
  iframe.style.bottom = '0';
  iframe.style.width = '0';
  iframe.style.height = '0';
  iframe.style.border = '0';
  document.body.appendChild(iframe);

  const doc = iframe.contentWindow?.document;
  if (!doc) {
    toast.error('Failed to generate PDF');
    return;
  }
  
  doc.open();
  doc.write(htmlContent);
  doc.close();

  // Clean up iframe after print dialog finishes
  iframe.contentWindow?.addEventListener('afterprint', () => {
    if (document.body.contains(iframe)) {
      document.body.removeChild(iframe);
    }
  });
}

const sourceTypeConfig: Record<string, { icon: React.ElementType; label: string; color: string }> = {
  pdf: { icon: FileText, label: 'PDF', color: 'bg-blue-500/10 text-blue-600' },
  url: { icon: LinkIcon, label: 'Article', color: 'bg-green-500/10 text-green-600' },
  topic: { icon: SquarePen, label: 'Topic', color: 'bg-purple-500/10 text-purple-600' },
}

export default function MaterialDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  const [material, setMaterial] = useState<Material | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isRenaming, setIsRenaming] = useState(false)
  const [renameInput, setRenameInput] = useState('')
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)
  const [isGeneratingQuiz, setIsGeneratingQuiz] = useState(false)


  useEffect(() => {
    if (id.startsWith('temp-')) {
      setIsLoading(false)
      return
    }
    const loadMaterial = async () => {
      // 1. Try cached_materials for instant display
      const stored = localStorage.getItem('cached_materials')
      if (stored) {
        try {
          const materials: Material[] = JSON.parse(stored)
          const found = materials.find((m) => m.id === id)
          if (found) setMaterial(found)
        } catch { }
      }
      // 2. Also fetch fresh data from API to get the real title/status
      try {
        const fresh = await materialsAPI.get(id)
        if (fresh) setMaterial(fresh)
      } catch {
        // API unavailable, localStorage version is fine
      }
      setIsLoading(false)
    }
    loadMaterial()
  }, [id])

  const handleRename = async () => {
    if (!renameInput.trim() || !material) return
    try {
      await materialsAPI.rename(material.id, renameInput.trim())
      setMaterial({ ...material, title: renameInput.trim() })
      const cached = localStorage.getItem('cached_materials')
      if (cached) {
        const materials: Material[] = JSON.parse(cached)
        const updated = materials.map((m) =>
          m.id === material.id ? { ...m, title: renameInput.trim() } : m
        )
        localStorage.setItem('cached_materials', JSON.stringify(updated))
      }
      toast.success('Material renamed!')
    } catch {
      toast.error('Failed to rename material')
    } finally {
      setIsRenaming(false)
      setRenameInput('')
    }
  }

  const handleDeleteMaterial = async () => {
    if (!material) return
    setIsDeleting(true)
    try {
      if (!material.id.startsWith('temp-') && !material.id.startsWith('topic-')) {
        await materialsAPI.delete(material.id)
      }
      const cached = localStorage.getItem('cached_materials')
      if (cached) {
        const materials: Material[] = JSON.parse(cached)
        localStorage.setItem('cached_materials', JSON.stringify(materials.filter((m) => m.id !== material.id)))
      }
      toast.success('Material deleted')
      router.push('/dashboard')
    } catch {
      toast.error('Failed to delete material')
    } finally {
      setIsDeleting(false)
      setIsDeleteDialogOpen(false)
    }
  }

  const sourceType = sourceTypeConfig[material?.source_type ?? 'pdf'] || sourceTypeConfig.pdf
  const SourceIcon = sourceType.icon
  const isTopic = material?.source_type === 'topic'
  const requestedTab = searchParams.get('tab')
  const normalizedRequestedTab =
    requestedTab === 'summary' || requestedTab === 'chat' || requestedTab === 'quiz'
      ? requestedTab
      : null
  const resolvedTab = normalizedRequestedTab ?? 'summary'

  const handleTabChange = (value: string) => {
    if (value === resolvedTab) return
    router.replace(`${pathname}?tab=${value}`, { scroll: false })
  }

  useEffect(() => {
    if (!pathname) return
    if (requestedTab === resolvedTab) return
    router.replace(`${pathname}?tab=${resolvedTab}`, { scroll: false })
  }, [pathname, requestedTab, resolvedTab, router])

  if (isLoading && !material) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!material) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-foreground mb-2">Material not found</h2>
          <Button onClick={() => router.push('/dashboard')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <nav className="border-b border-border/50 bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/dashboard" className="flex items-center gap-2">
              <Logo />
            </Link>
            <div className="flex items-center gap-3">
              <UsageStats />
              <UserDropdown />
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push('/dashboard')}
            className="mb-4 -ml-2"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Materials
          </Button>

          <div className="flex items-start gap-4">
            <div className={`p-3 rounded-xl ${sourceType.color}`}>
              <SourceIcon className="w-6 h-6" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                {isRenaming ? (
                  <div className="flex items-center gap-2">
                    <Input
                      value={renameInput}
                      onChange={(e) => setRenameInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRename()
                        if (e.key === 'Escape') { setIsRenaming(false); setRenameInput('') }
                      }}
                      maxLength={200}
                      className="text-xl font-bold h-10 w-64"
                      autoFocus
                    />
                    <Button size="sm" onClick={handleRename} disabled={!renameInput.trim()}>
                      Save
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => { setIsRenaming(false); setRenameInput('') }}>
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <>
                    <h1 className="text-2xl sm:text-3xl font-bold text-foreground truncate">
                      {material.title}
                    </h1>
                    {!isTopic && (
                      <button
                        onClick={() => { setIsRenaming(true); setRenameInput(material.title) }}
                        className="p-1.5 rounded-lg hover:bg-muted transition-colors flex-shrink-0"
                        title="Rename"
                      >
                        <Pencil className="w-4 h-4 text-muted-foreground hover:text-foreground" />
                      </button>
                    )}
                    <button
                      onClick={() => setIsDeleteDialogOpen(true)}
                      className="p-1.5 rounded-lg hover:bg-destructive/10 transition-colors flex-shrink-0"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
                    </button>
                  </>
                )}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Badge variant="secondary" className={sourceType.color}>
                  {sourceType.label}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  Added {new Date(material.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>
        </div>

        <Tabs value={resolvedTab} onValueChange={handleTabChange} className="mt-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="summary" className="gap-2 hover:bg-primary/5 transition-colors">
              <Sparkles className="h-4 w-4" />
              <span className="hidden sm:inline">Summary</span>
            </TabsTrigger>
            <TabsTrigger value="chat" className="gap-2 hover:bg-primary/5 transition-colors">
              <MessageSquare className="h-4 w-4" />
              <span className="hidden sm:inline">Chat</span>
            </TabsTrigger>
            <TabsTrigger value="quiz" className="gap-2 hover:bg-primary/5 transition-colors">
              <Brain className="h-4 w-4" />
              <span className="hidden sm:inline">Quiz</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="summary" forceMount className="mt-6">
            <SummaryTab
              materialId={id}
              sourceType={material.source_type}
              materialTitle={material.title}
              isGenerating={isGeneratingSummary}
              setIsGenerating={setIsGeneratingSummary}
              onTabChange={handleTabChange}
            />
          </TabsContent>

          <TabsContent value="chat" forceMount className="mt-6">
            <ChatTab materialId={id} sourceType={material.source_type} topic={material.topic} materialTitle={material.title} />
          </TabsContent>

          <TabsContent value="quiz" forceMount className="mt-6">
            <QuizTab
              materialId={id}
              sourceType={material.source_type}
              topic={material.topic}
              materialTitle={material.title}
              isGenerating={isGeneratingQuiz}
              setIsGenerating={setIsGeneratingQuiz}
            />
          </TabsContent>
        </Tabs>
      </main>

      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Material</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this material? All associated chats, summaries, and quizzes will be permanently removed. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={(e) => {
                e.preventDefault()
                handleDeleteMaterial()
              }}
              disabled={isDeleting}
            >
              {isDeleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

function SummaryTab({ materialId, sourceType, materialTitle, isGenerating, setIsGenerating, onTabChange }: {
  materialId: string
  sourceType: string
  materialTitle: string
  isGenerating: boolean
  setIsGenerating: (v: boolean) => void
  onTabChange: (value: string) => void
}) {
  const [summary, setSummary] = useState<string | null>(null)
  const [isLoadingExisting, setIsLoadingExisting] = useState(true)
  const { updateUser } = useAuth()

  const refreshUsage = async () => {
    try {
      const data = await usageAPI.getUsage()
      updateUser({ usage: data })
      localStorage.setItem('usage_cache', JSON.stringify(data))
    } catch (err) {
      console.error('Failed to refresh usage:', err)
    }
  }

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isMounted = useRef(true)

  useEffect(() => {
    isMounted.current = true
    return () => { isMounted.current = false }
  }, [])

  const startSummaryPoller = () => {
    if (pollingRef.current) return
    pollingRef.current = setInterval(async () => {
      try {
        const result = await materialsAPI.getSummary(materialId)
        if (result.summary && isMounted.current) {
          setSummary(result.summary)
          setIsLoadingExisting(false)
          setIsGenerating(false)
          localStorage.removeItem(`generating_summary_${materialId}`)
          clearInterval(pollingRef.current!)
          pollingRef.current = null
          refreshUsage()
        }
      } catch { }
    }, 4000)
  }

  useEffect(() => {
    const checkStatus = async () => {
      // 1. Check if we should be generating (from localStorage)
      const wasGenerating = localStorage.getItem(`generating_summary_${materialId}`) === 'true'
      if (wasGenerating) {
        setIsGenerating(true)
        startSummaryPoller()
      }

      // 2. Load existing summary
      try {
        const result = await materialsAPI.getSummary(materialId)
        if (result.summary) {
          setSummary(result.summary)
          if (wasGenerating) {
            setIsGenerating(false)
            localStorage.removeItem(`generating_summary_${materialId}`)
            if (pollingRef.current) {
              clearInterval(pollingRef.current)
              pollingRef.current = null
            }
          }
        }
      } catch {
        // no existing summary
      } finally {
        if (isMounted.current) setIsLoadingExisting(false)
      }
    }
    checkStatus()

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [materialId, sourceType])

  const generateSummary = async () => {
    setIsLoadingExisting(false)
    setIsGenerating(true)
    localStorage.setItem(`generating_summary_${materialId}`, 'true')
    startSummaryPoller() // Start polling in case we navigate away and back

    try {
      const result = await materialsAPI.summarize(materialId)
      if (isMounted.current) {
        setSummary(result.summary)
        setIsLoadingExisting(false)
        toast.success('Summary generated!')
        refreshUsage()
      }
    } catch {
      toast.error('Failed to generate summary. Please try again.')
    } finally {
      if (isMounted.current) {
        setIsGenerating(false)
        localStorage.removeItem(`generating_summary_${materialId}`)
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }
    }
  }

  const handleExportPDF = () => {
    if (!summary) return
    let subIndex = 0
    const html = summary
      .split('\n')
      .map((p) => {
        const trimmed = p.trim()
        if (!trimmed) return ''

        // Headers
        const mainMatch = trimmed.match(/^\[{2,4}###\s*(.+?)\s*###\]{2,4}$/)
        const subMatch = trimmed.match(/^\[{2,4}>>>\s*(.+?)\s*<<<\]{2,4}$/)

        if (mainMatch || trimmed.startsWith('###')) {
          const content = mainMatch ? mainMatch[1] : trimmed.replace(/^#+\s+/, '')
          return `<h2 class="summary-h2">${content}</h2>`
        }
        if (subMatch || trimmed.startsWith('####')) {
          subIndex++
          const content = subMatch ? subMatch[1] : trimmed.replace(/^#+\s+/, '')
          const hasNumbering = /^\d+[\.\-]/.test(content)
          return `<h4 class="summary-h4">${hasNumbering ? content : `${subIndex}. ${content}`}</h4>`
        }
        if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
          return `<h2 class="summary-h2">${trimmed.replace(/\*\*/g, '')}</h2>`
        }

        // Convert numbered lists to bulleted lists to avoid double numbering with subheaders
        if (trimmed.match(/^\d+\./)) {
          // Remove the number and any following bullets/spaces
          let content = trimmed.replace(/^\d+\.\s*(?:[•\-\–\—\+]\s*|\*(?!\*)\s*)?/, '')
          if (content.includes('**')) {
            content = content.replace(/\*\*(.*?)\*\*/g, '<span class="topic-bold">$1</span>')
          }
          return `<div class="summary-list-item">${content}</div>`
        }

        // List item with potential bold topic
        if (trimmed.startsWith('- ')) {
          let content = trimmed.slice(2)
          if (content.includes('**')) {
            content = content.replace(/\*\*(.*?)\*\*/g, '<span class="topic-bold">$1</span>')
          }
          return `<div class="summary-list-item">${content}</div>`
        }

        // Paragraph with potential bold
        let pContent = trimmed
        if (pContent.includes('**')) {
          pContent = pContent.replace(/\*\*(.*?)\*\*/g, '<span class="topic-bold">$1</span>')
        }
        return `<p class="summary-p">${pContent}</p>`
      })
      .join('')
    printAsPDF(`Summary - ${materialTitle}`, html)
  }

  if (isLoadingExisting && !summary && !isGenerating) {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="text-center py-12">
          <CardContent>
            <Loader2 className="w-8 h-8 mx-auto animate-spin text-muted-foreground" />
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
      {isGenerating ? (
        <Card className="text-center py-16">
          <CardContent className="flex flex-col items-center gap-6">
            <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xl font-semibold text-foreground">Generating your summary</span>
              <span className="text-muted-foreground">This might take a while...</span>
            </div>
          </CardContent>
        </Card>
      ) : !summary ? (
        <Card className="text-center py-12">
          <CardContent>
            <div className="w-16 h-16 mx-auto bg-primary/10 rounded-2xl flex items-center justify-center mb-6">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-semibold text-foreground mb-2">Generate AI Summary</h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              Let AI analyze your material and create a comprehensive summary with key points and insights.
            </p>
            <Button onClick={generateSummary} disabled={isGenerating} size="lg">
              <Sparkles className="mr-2 h-4 w-4" />Generate Summary
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>AI-Generated Summary</CardTitle>
              <CardDescription>Key points and insights from your material</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleExportPDF}>
              <Printer className="mr-2 h-4 w-4" />
              Export PDF
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-8">
              {(() => {
                const lines = summary.split('\n')
                const sections: { title: string; level: number; content: string[] }[] = []
                let currentSection: { title: string; level: number; content: string[] } | null = null

                let lastHeader = ''
                lines.forEach(line => {
                  const trimmed = line.trim()
                  if (!trimmed) return

                  // Parse new unique markers
                  const mainHeaderMatch = trimmed.match(/^\[{2,4}###\s*(.+?)\s*###\]{2,4}$/)
                  const subHeaderMatch = trimmed.match(/^\[{2,4}>>>\s*(.+?)\s*<<<\]{2,4}$/)

                  // Legacy/Fallback matches
                  const legacyHeaderMatch = line.match(/^(#{3,4})\s+(.+)/)
                  const boldHeaderMatch = trimmed.startsWith('**') && trimmed.endsWith('**') && trimmed.length < 100
                  const numberedHeaderMatch = trimmed.match(/^\d+\s+(.+)/)

                  if (mainHeaderMatch || subHeaderMatch || legacyHeaderMatch || boldHeaderMatch || numberedHeaderMatch) {
                    const level = mainHeaderMatch ? 3 : (subHeaderMatch ? 4 : (legacyHeaderMatch ? legacyHeaderMatch[1].length : 3))
                    const title = mainHeaderMatch
                      ? mainHeaderMatch[1].trim()
                      : subHeaderMatch
                        ? subHeaderMatch[1].trim()
                        : legacyHeaderMatch
                          ? legacyHeaderMatch[2].replace(/\*+/g, '').trim()
                          : boldHeaderMatch
                            ? trimmed.replace(/\*\*/g, '').trim()
                            : numberedHeaderMatch
                              ? numberedHeaderMatch[1].trim()
                              : trimmed
                    const normalized = title.toLowerCase()
                    if (normalized && normalized === lastHeader) {
                      return
                    }
                    lastHeader = normalized
                    currentSection = { title, level, content: [] }
                    sections.push(currentSection)
                  } else if (currentSection) {
                    currentSection.content.push(line)
                  } else if (sections.length === 0) {
                    // Handle text before any header
                    currentSection = { title: 'Overview', level: 3, content: [line] }
                    sections.push(currentSection)
                  } else {
                    sections[sections.length - 1].content.push(line)
                  }
                })

                let subHeaderIndex = 0
                return sections.map((section, idx) => {
                  const titleLower = section.title.toLowerCase()
                  const isMajor = section.level === 3 || titleLower.includes('summary') || titleLower.includes('overview')
                  const tightenWidth = titleLower.includes('takeaway') || titleLower.includes('key topic')

                  if (isMajor) {
                    return (
                      <div key={idx} className="py-6">
                        {/* Centered Header with Lines */}
                        <div className="flex items-center gap-4 mb-8">
                          <div className="h-px flex-1 bg-gradient-to-r from-transparent to-primary/50" />
                          <h3 className="text-2xl font-bold text-primary px-4 whitespace-nowrap tracking-wide uppercase">
                            {section.title}
                          </h3>
                          <div className="h-px flex-1 bg-gradient-to-l from-transparent to-primary/50" />
                        </div>

                        <div className={`space-y-4 ${tightenWidth ? 'max-w-2xl' : 'max-w-3xl'} mx-auto px-4`}>
                          {section.content.map((line, lIdx) => (
                            <p dir="auto" key={lIdx} className={`text-[19.5px] text-foreground leading-relaxed ${tightenWidth ? 'text-left' : 'text-center'}`}>
                              {line.split('**').map((part, i) =>
                                i % 2 === 1 ? <strong key={i} className="text-foreground">{part}</strong> : part
                              )}
                            </p>
                          ))}
                        </div>
                      </div>
                    )
                  }

                  // Sub-header as Collapsible Box (Accordion Style)
                  subHeaderIndex++
                  return (
                    <div key={idx} className="max-w-4xl mx-auto px-4 mb-4">
                      <Collapsible className="group">
                        <CollapsibleTrigger className="flex w-full items-center justify-between p-4 rounded-xl border border-border bg-card/50 hover:bg-muted/50 transition-all text-left shadow-sm">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center justify-center w-6 h-6 rounded-lg bg-primary/10 text-primary group-data-[state=open]:rotate-180 transition-transform">
                              <ChevronDown className="h-4 w-4" />
                            </div>
                            <span className="text-lg font-bold text-foreground">{subHeaderIndex}. {section.title}</span>
                          </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent className="px-6 py-4 space-y-4 border-x border-b rounded-b-xl bg-card/30">
                          {section.content.map((line, lIdx) => {
                            const tLine = line.trim()
                            if (tLine.startsWith('- ')) {
                              const content = tLine.replace(/^- /, '')
                              return (
                                <div key={lIdx} className="flex gap-3 items-start text-lg text-foreground leading-relaxed">
                                  <div className="w-1.5 h-1.5 rounded-full bg-primary/40 mt-2 flex-shrink-0" />
                                  <span>
                                    {content.split('**').map((part, i) =>
                                      i % 2 === 1 ? <strong key={i} className="text-foreground">{part}</strong> : part
                                    )}
                                  </span>
                                </div>
                              )
                            }
                            return (
                              <p dir="auto" key={lIdx} className="text-lg text-foreground leading-relaxed">
                                {line.split('**').map((part, i) =>
                                  i % 2 === 1 ? <strong key={i} className="text-foreground">{part}</strong> : part
                                )}
                              </p>
                            )
                          })}
                        </CollapsibleContent>
                      </Collapsible>
                    </div>
                  )
                })
              })()}
            </div>
          </CardContent>
        </Card>
      )}
    </motion.div>
  )
}

function ChatTab({ materialId, sourceType, topic, materialTitle }: {
  materialId: string
  sourceType: string
  topic?: string
  materialTitle: string
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSessionsLoading, setIsSessionsLoading] = useState(true)
  const [isCreatingSession, setIsCreatingSession] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null)
  const [renameSessionInput, setRenameSessionInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const lastUserMessageRef = useRef<HTMLDivElement | null>(null)
  const shouldScrollToUserRef = useRef(false)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isMounted = useRef(true)

  useEffect(() => {
    isMounted.current = true
    return () => { isMounted.current = false }
  }, [])

  const startChatPoller = (sid: string) => {
    if (pollingRef.current) return
    pollingRef.current = setInterval(async () => {
      try {
        const data = await tutorAPI.getSessionMessages(sid)
        const formatted: ChatMessage[] = data.map((m: any) => ({
          role: m.role,
          content: m.content
        }))

        // If the last message is from assistant, we are done
        if (formatted.length > 0 && formatted[formatted.length - 1].role === 'assistant') {
          if (isMounted.current) {
            setMessages(formatted)
            setIsLoading(false)
            localStorage.removeItem(`waiting_chat_${sid}`)
            clearInterval(pollingRef.current!)
            pollingRef.current = null
          }
        }
      } catch { }
    }, 3000)
  }

  const isInitializingRef = useRef(false)

  useEffect(() => {
    const loadSessions = async () => {
      if (isInitializingRef.current) return
      isInitializingRef.current = true
      
      try {
        const data = await tutorAPI.listSessions(materialId)
        if (isMounted.current) setSessions(data)
        
        if (data.length > 0) {
          if (isMounted.current) setCurrentSessionId(data[0].id)
        } else {
          try {
            const newSession = await tutorAPI.createSession(materialId, 'Untitled Chat')
            if (isMounted.current) {
              setSessions([newSession])
              setCurrentSessionId(newSession.id)
              setMessages([])
            }
          } catch (createErr) {
            console.error('Failed to auto-create session', createErr)
          }
        }
      } catch (err) {
        console.error('Failed to load sessions', err)
      } finally {
        if (isMounted.current) setIsSessionsLoading(false)
        isInitializingRef.current = false
      }
    }
    loadSessions()
  }, [materialId])

  useEffect(() => {
    if (currentSessionId) {
      const loadMessages = async () => {
        const sid = currentSessionId!

        // Check if we were waiting for a response in this session
        const wasWaiting = localStorage.getItem(`waiting_chat_${sid}`) === 'true'
        if (wasWaiting) {
          setIsLoading(true)
          startChatPoller(sid)
        }

        try {
          const data = await tutorAPI.getSessionMessages(sid)
          const formatted: ChatMessage[] = data.map((m: any) => ({
            role: m.role,
            content: m.content
          }))

          if (isMounted.current) {
            setMessages(formatted)

            // If we were waiting but the assistant message is already here
            if (wasWaiting && formatted.length > 0 && formatted[formatted.length - 1].role === 'assistant') {
              setIsLoading(false)
              localStorage.removeItem(`waiting_chat_${sid}`)
              if (pollingRef.current) {
                clearInterval(pollingRef.current)
                pollingRef.current = null
              }
            }
          }
        } catch (err) {
          console.error('Failed to load messages', err)
        }
      }
      loadMessages()
    } else {
      setMessages([])
    }

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [currentSessionId])

  const scrollToLastUser = () => {
    if (lastUserMessageRef.current) {
      lastUserMessageRef.current.scrollIntoView({ block: 'start', behavior: 'auto' })
    }
  }

  useEffect(() => {
    if (shouldScrollToUserRef.current) {
      scrollToLastUser()
      shouldScrollToUserRef.current = false
    }
  }, [messages])

  const createNewSession = async () => {
    setIsCreatingSession(true)
    try {
      const newSession = await tutorAPI.createSession(materialId, 'Untitled Chat')
      setSessions([newSession, ...sessions])
      setCurrentSessionId(newSession.id)
      setMessages([])
    } catch (err) {
      toast.error('Failed to create new session')
    } finally {
      setIsCreatingSession(false)
    }
  }

  const deleteSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await tutorAPI.deleteSession(sid)
      const updated = sessions.filter(s => s.id !== sid)
      setSessions(updated)
      if (currentSessionId === sid) {
        setCurrentSessionId(updated.length > 0 ? updated[0].id : null)
      }
      localStorage.removeItem(`waiting_chat_${sid}`)
      toast.success('Session deleted')
    } catch (err) {
      toast.error('Failed to delete session')
    }
  }

  const renameSession = async (sid: string, title: string) => {
    if (!title.trim()) return
    try {
      await tutorAPI.renameSession(sid, title.trim())
      setSessions(prev => prev.map(s => s.id === sid ? { ...s, title: title.trim() } : s))
      toast.success('Session renamed')
    } catch {
      toast.error('Failed to rename session')
    } finally {
      setRenamingSessionId(null)
      setRenameSessionInput('')
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    let sessionId = currentSessionId
    if (!sessionId) {
      try {
        const newSession = await tutorAPI.createSession(materialId, 'Untitled Chat')
        setSessions([newSession])
        setCurrentSessionId(newSession.id)
        sessionId = newSession.id
      } catch (err) {
        toast.error('Failed to initialize chat')
        return
      }
    }

    const currentInput = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: currentInput }])
    setIsLoading(true)
    localStorage.setItem(`waiting_chat_${sessionId}`, 'true')
    startChatPoller(sessionId) // Start polling in case we navigate away
    shouldScrollToUserRef.current = true

    try {
      const data = await tutorAPI.ask(currentInput, sourceType, materialId, sessionId!)
      if (isMounted.current && currentSessionId === sessionId) {
        setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }])

        if (sessions.find(s => s.id === sessionId)?.title === 'Untitled Chat') {
          try {
            const titleData = await tutorAPI.extractTitle(sessionId!, currentInput)
            setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title: titleData.title } : s))
          } catch { }
        }
      }
    } catch {
      if (isMounted.current) toast.error('Failed to get response')
    } finally {
      if (isMounted.current) {
        setIsLoading(false)
        localStorage.removeItem(`waiting_chat_${sessionId}`)
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }
    }
  }

  const handleExportPDF = () => {
    if (messages.length === 0) return
    const currentTitle = sessions.find(s => s.id === currentSessionId)?.title || 'Chat'
    const escapeHtml = (value: string) =>
      value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')

    const formatInline = (value: string) =>
      escapeHtml(value).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')

    const html = messages.map(m => {
      const lines = m.content.split('\n')
      const parts = lines.map((line) => {
        const trimmed = line.trim()
        if (!trimmed) return '<div style="height:6px"></div>'

        const headerMatch = trimmed.match(/^(#{1,3})\s*(.+)$/)
        if (headerMatch) {
          const level = headerMatch[1].length
          const text = formatInline(headerMatch[2])
          if (level === 1) return `<h2 style="margin:12px 0 6px;font-size:18px;font-weight:700;">${text}</h2>`
          if (level === 2) return `<h3 style="margin:10px 0 6px;font-size:16px;font-weight:700;">${text}</h3>`
          return `<h4 style="margin:8px 0 4px;font-size:14px;font-weight:700;">${text}</h4>`
        }

        const bulletMatch = trimmed.match(/^[-*+]\s+(.*)$/)
        if (bulletMatch) {
          return `<div style="margin:4px 0 4px 16px;">• ${formatInline(bulletMatch[1])}</div>`
        }

        const numberMatch = trimmed.match(/^(\d+\.)\s+(.*)$/)
        if (numberMatch) {
          return `<div style="margin:4px 0 4px 16px;">${formatInline(numberMatch[1])} ${formatInline(numberMatch[2])}</div>`
        }

        return `<div style="margin:4px 0;">${formatInline(trimmed)}</div>`
      }).join('')

      return `
      <div class="chat-msg ${m.role === 'user' ? 'chat-user' : 'chat-ai'}">
        <span class="chat-role">${m.role}</span>
        <div>${parts}</div>
      </div>
    `}).join('')
    printAsPDF(`Chat Session - ${currentTitle}`, html)
  }

  return (
    <div className="h-[750px] flex flex-col overflow-hidden border border-border/50 shadow-sm bg-card rounded-xl">
      {/* Header */}
      <div className="px-4 border-b bg-slate-50 dark:bg-[#0e0e0e] h-20 flex items-center justify-between shrink-0 group">
        <div className="flex items-center gap-3 min-w-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="h-8 w-8 p-0 hidden md:flex flex-shrink-0 hover:bg-muted"
            title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
          >
            {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
          </Button>
          <Badge variant="secondary" className="bg-primary/10 text-primary text-[10px] h-5 flex-shrink-0 px-2 font-bold">LIVE</Badge>
          <div className="flex items-center gap-1 min-w-0">
            <span className="text-base font-semibold truncate text-foreground">
              {(sessions.find(s => s.id === currentSessionId)?.title || 'Untitled Chat').slice(0, 50)}
            </span>
            {currentSessionId && (
              <>
                <button
                  onClick={() => {
                    const s = sessions.find(s => s.id === currentSessionId)
                    if (s) { setRenamingSessionId(s.id); setRenameSessionInput(s.title) }
                  }}
                  className="p-1 rounded-lg hover:bg-muted transition-colors flex-shrink-0 opacity-0 group-hover:opacity-100"
                  title="Rename session"
                >
                  <Pencil className="w-3.5 h-3.5 text-muted-foreground hover:text-foreground" />
                </button>
                <button
                  onClick={(e) => deleteSession(currentSessionId, e as any)}
                  className="p-1 rounded-lg hover:bg-destructive/10 transition-colors flex-shrink-0 opacity-0 group-hover:opacity-100"
                  title="Delete session"
                >
                  <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive" />
                </button>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Button variant="ghost" size="sm" onClick={handleExportPDF} disabled={messages.length === 0} className="h-9 w-9 p-0 hover:bg-muted">
            <Printer className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Sidebar */}
        {sidebarOpen && (
          <div className="hidden md:flex flex-col w-60 border-r bg-slate-50 dark:bg-[#0e0e0e] min-h-0">
            <div className="p-4 border-b">
              <Button
                onClick={createNewSession}
                className="w-full justify-start gap-2 bg-background hover:bg-muted transition-colors"
                variant="outline"
                disabled={isCreatingSession}
              >
                <Plus className="h-4 w-4" />
                New Chat
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto px-3" style={{ scrollbarGutter: 'stable' }}>
              <div className="space-y-1 py-3">
                {isSessionsLoading ? (
                  <div className="flex justify-center py-4">
                    <Loader2 className="animate-spin h-5 w-5 text-muted-foreground" />
                  </div>
                ) : sessions.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => {
                      if (renamingSessionId !== s.id) setCurrentSessionId(s.id)
                    }}
                    className={`group flex items-start justify-between p-2 rounded-lg cursor-pointer transition-all ${currentSessionId === s.id ? 'bg-primary/10 border-primary/20 border shadow-sm' : 'hover:bg-muted border border-transparent'
                      }`}
                  >
                    {renamingSessionId === s.id ? (
                      <div className="flex items-center gap-1 w-full" onClick={(e) => e.stopPropagation()}>
                        <Input
                          value={renameSessionInput}
                          onChange={(e) => setRenameSessionInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') renameSession(s.id, renameSessionInput)
                            if (e.key === 'Escape') { setRenamingSessionId(null); setRenameSessionInput('') }
                          }}
                          onBlur={() => { setRenamingSessionId(null); setRenameSessionInput('') }}
                          maxLength={200}
                          className="text-sm h-8 py-1 px-2"
                          autoFocus
                        />
                      </div>
                    ) : (
                      <>
                        <div className="flex items-start gap-2 overflow-hidden min-w-0 flex-1">
                          <MessageSquare className={`h-4 w-4 flex-shrink-0 mt-0.5 ${currentSessionId === s.id ? 'text-primary' : 'text-muted-foreground'}`} />
                          <span className={`text-sm whitespace-normal break-words leading-tight ${currentSessionId === s.id ? 'font-semibold text-primary' : 'text-foreground'}`}>
                            {(s.title || 'Untitled Chat').slice(0, 50)}
                          </span>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                            <button className="opacity-0 group-hover:opacity-100 p-1 hover:text-primary transition-all flex-shrink-0">
                              <MoreHorizontal className="h-3.5 w-3.5" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                            <DropdownMenuItem
                              className="cursor-pointer"
                              onClick={() => { setRenamingSessionId(s.id); setRenameSessionInput(s.title) }}
                            >
                              <Pencil className="mr-2 h-4 w-4" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="cursor-pointer text-destructive focus:text-destructive"
                              onClick={(e) => deleteSession(s.id, e as any)}
                            >
                              <Trash2 className="mr-2 h-4 w-4" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Chat Content */}
        <div className="flex-1 flex flex-col min-w-0 relative">
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-6"
            style={{ scrollbarGutter: 'stable' }}
          >
            {!currentSessionId ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 mt-20">
                <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-4">
                  <MessageSquarePlus className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">No active session</h3>
                <p className="text-sm text-muted-foreground max-w-xs mb-6">
                  Start a new conversation to begin discussing this material.
                </p>
                <Button onClick={createNewSession} disabled={isCreatingSession}>
                  <Plus className="mr-2 h-4 w-4" /> New Session
                </Button>
              </div>
            ) : messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 mt-20">
                <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-4">
                  <MessageSquare className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Start a conversation</h3>
                <p className="text-sm text-muted-foreground max-w-xs">
                  Ask questions about the material, request clarifications, or explore related topics.
                </p>
              </div>
            ) : (
              <div className="space-y-6 max-w-3xl mx-auto w-full">
                {messages.map((m, i) => (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={i}
                    className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    ref={m.role === 'user' ? lastUserMessageRef : undefined}
                  >
                    <div className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${m.role === 'user'
                        ? 'bg-primary text-primary-foreground rounded-tr-none'
                        : 'bg-card border border-border/50 text-foreground rounded-tl-none'
                      }`}>
                      {m.role === 'user' ? (
                        <p dir="auto" className="text-[15.25px] leading-relaxed whitespace-pre-wrap">{m.content}</p>
                      ) : (
                        <div className="text-[16.25px] leading-relaxed [&_strong]:font-semibold [&_li]:mb-1">
                          {renderMarkdown(m.content)}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-card border border-border/50 rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <span className="text-sm text-muted-foreground">Assistant is thinking...</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-3 border-t bg-slate-50 dark:bg-[#0e0e0e]">
            <form
              onSubmit={(e) => { e.preventDefault(); sendMessage() }}
              className="flex gap-2 max-w-3xl mx-auto"
            >
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={currentSessionId ? "Ask anything..." : "Create a session to start chatting"}
                disabled={!currentSessionId}
                className="flex-1 rounded-xl bg-muted/30 border-border/50 focus-visible:ring-primary h-10"
              />
              <Button type="submit" disabled={isLoading || !input.trim() || !currentSessionId} size="icon" className="h-10 w-10 rounded-xl shadow-md">
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

function QuizTab({ materialId, sourceType, topic, materialTitle, isGenerating, setIsGenerating }: {
  materialId: string
  sourceType: string
  topic?: string
  materialTitle: string
  isGenerating: boolean
  setIsGenerating: (v: boolean) => void
}) {
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium')
  const [mcqCount, setMcqCount] = useState(10)
  const [tfCount, setTfCount] = useState(5)
  const [mcqCountInput, setMcqCountInput] = useState('10')
  const [tfCountInput, setTfCountInput] = useState('5')
  const [questions, setQuestions] = useState<QuizQuestion[]>([])
  const [quizId, setQuizId] = useState<string | null>(null)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [result, setResult] = useState<QuizResult | null>(null)
  const [isLoadingExistingQuiz, setIsLoadingExistingQuiz] = useState(true)
  const { updateUser } = useAuth()

  const refreshUsage = async () => {
    try {
      const data = await usageAPI.getUsage()
      updateUser({ usage: data })
      localStorage.setItem('usage_cache', JSON.stringify(data))
    } catch (err) {
      console.error('Failed to refresh usage:', err)
    }
  }

  const clampCount = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isMounted = useRef(true)
  const pendingQuizBaselineIdRef = useRef<string | null>(null)
  const getQuizDraftKey = (id: string) => `quiz_draft_${materialId}_${id}`

  const clearQuizDraft = (id?: string | null) => {
    if (!id) return
    localStorage.removeItem(getQuizDraftKey(id))
  }

  const readQuizDraft = (id: string, quizQuestions: QuizQuestion[]) => {
    try {
      const raw = localStorage.getItem(getQuizDraftKey(id))
      if (!raw) return {}
      const parsed = JSON.parse(raw) as { answers?: Record<string, string> }
      const validQuestionIds = new Set(quizQuestions.map((q) => q.id))
      return Object.fromEntries(
        Object.entries(parsed.answers || {}).filter(
          ([questionId, value]) => validQuestionIds.has(questionId) && typeof value === 'string' && value.trim()
        )
      ) as Record<string, string>
    } catch {
      return {}
    }
  }

  const stopQuizGenerating = () => {
    setIsGenerating(false)
    pendingQuizBaselineIdRef.current = null
    localStorage.removeItem(`generating_quiz_${materialId}`)
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }

  useEffect(() => { setMcqCountInput(String(mcqCount)) }, [mcqCount])
  useEffect(() => { setTfCountInput(String(tfCount)) }, [tfCount])

  // ── Parse quiz data from backend format ──────────────
  const parseQuizData = (rawQuiz: any): QuizQuestion[] => {
    const mcqQuestions: QuizQuestion[] = (rawQuiz.mcq || []).map((q: any, i: number) => ({
      id: `q-mcq-${i}`,
      type: 'mcq' as const,
      question: q.question || '',
      options: q.options || [],
      correct_answer: q.answer || '',
      explanation: q.explanation || '',
    }))

    const tfQuestions: QuizQuestion[] = (rawQuiz.tf || []).map((q: any, i: number) => ({
      id: `q-tf-${i}`,
      type: 'true_false' as const,
      question: q.question || '',
      options: undefined,
      correct_answer: q.answer || '',
      explanation: q.explanation || '',
    }))

    return [...mcqQuestions, ...tfQuestions]
  }

  useEffect(() => {
    isMounted.current = true
    return () => { isMounted.current = false }
  }, [])

  const startQuizPoller = (baselineQuizId?: string | null) => {
    if (pollingRef.current) return
    pendingQuizBaselineIdRef.current = baselineQuizId ?? null
    pollingRef.current = setInterval(async () => {
      try {
        const quizzes = await quizAPI.list(materialId)
        if (quizzes.length > 0 && isMounted.current) {
          const latest = quizzes[quizzes.length - 1]
          if (pendingQuizBaselineIdRef.current && latest.id === pendingQuizBaselineIdRef.current) {
            return
          }
          setQuizId(latest.id)
          const rawQuiz = latest.quiz_data || latest.quiz
          if (rawQuiz) {
            const formatted = parseQuizData(rawQuiz)
            if (formatted.length > 0) {
              setQuestions(formatted)
              setAnswers(readQuizDraft(latest.id, formatted))
            }
          }
          setResult(null)
          stopQuizGenerating()
          refreshUsage()
        }
      } catch { }
    }, 5000)
  }

  // ── Load existing quiz on mount ──────────────────────
  useEffect(() => {
    const checkStatus = async () => {
      const wasGenerating = localStorage.getItem(`generating_quiz_${materialId}`) === 'true'
      setIsLoadingExistingQuiz(true)

      try {
        const quizzes = await quizAPI.list(materialId)
        const latest = quizzes.length > 0 ? quizzes[quizzes.length - 1] : null

        if (wasGenerating) {
          setIsGenerating(true)
          setQuestions([])
          setAnswers({})
          setResult(null)
          startQuizPoller(latest?.id ?? null)
        }

        if (latest) {
          const rawQuiz = latest.quiz_data || latest.quiz
          const formatted = rawQuiz ? parseQuizData(rawQuiz) : []
          let latestResult: QuizResult | null = null

          try {
            const results = await quizAPI.getResults(latest.id)
            if (results.length > 0 && !wasGenerating) {
              const lastResult = results[results.length - 1].results
              if (lastResult) latestResult = lastResult as QuizResult
            }
          } catch { }

          setQuizId(latest.id)
          if (!wasGenerating) {
            setQuestions(formatted)
            if (latestResult) {
              clearQuizDraft(latest.id)
              setAnswers({})
            } else {
              setAnswers(readQuizDraft(latest.id, formatted))
            }
            setResult(latestResult)
          }
        }
      } catch { }
      finally {
        if (isMounted.current) {
          setIsLoadingExistingQuiz(false)
        }
      }
    }
    checkStatus()

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [materialId, sourceType])

  // ── Generate quiz ────────────────────────────────────
  const generateQuiz = async () => {
    const baselineQuizId = quizId
    setIsGenerating(true)
    setQuestions([])
    setResult(null)
    setAnswers({})
    setQuizId(null)
    localStorage.setItem(`generating_quiz_${materialId}`, 'true')
    startQuizPoller(baselineQuizId)

    let generated = false

    try {
      const data = await quizAPI.generate(
        difficulty,
        mcqCount,
        tfCount,
        sourceType,
        materialId,
        topic,
      )
      if (isMounted.current) {
        generated = true
        setQuizId(data.quiz_id)
        const formatted = parseQuizData(data.quiz)
        if (formatted.length > 0) {
          setQuestions(formatted)
          setAnswers(readQuizDraft(data.quiz_id, formatted))
          stopQuizGenerating()
          toast.success('Quiz generated!')
          refreshUsage()
        } else {
          throw new Error('No questions returned')
        }
      }
    } catch {
      toast.error('Failed to generate quiz. Please try again.')
    } finally {
      if (isMounted.current && !generated) {
        stopQuizGenerating()
      }
    }
  }

  // ── Submit ───────────────────────────────────────────
  const submitQuiz = () => {
    const results: QuizResult['answers'] = questions.map((q) => ({
      question_id: q.id,
      user_answer: answers[q.id] || '',
      correct_answer: q.correct_answer,
      is_correct: (answers[q.id] || '').toLowerCase() === q.correct_answer.toLowerCase(),
    }))
    const correct = results.filter((r) => r.is_correct).length
    const resultData: QuizResult = {
      total: questions.length,
      correct,
      incorrect: questions.length - correct,
      score: Math.round((correct / questions.length) * 100),
      answers: results,
    }
    setResult(resultData)
    if (quizId) {
      clearQuizDraft(quizId)
      quizAPI.saveResult(quizId, resultData as unknown as Record<string, unknown>).catch(() => { })
    }
    toast.success(`Quiz completed! Score: ${resultData.score}%`)
  }

  const resetQuiz = () => {
    clearQuizDraft(quizId)
    setQuestions([])
    setAnswers({})
    setResult(null)
  }

  useEffect(() => {
    if (!quizId || questions.length === 0 || result) return

    const validQuestionIds = new Set(questions.map((q) => q.id))
    const filteredAnswers = Object.fromEntries(
      Object.entries(answers).filter(
        ([questionId, value]) => validQuestionIds.has(questionId) && typeof value === 'string' && value.trim()
      )
    )

    if (Object.keys(filteredAnswers).length === 0) {
      clearQuizDraft(quizId)
      return
    }

    localStorage.setItem(getQuizDraftKey(quizId), JSON.stringify({ answers: filteredAnswers }))
  }, [answers, questions, quizId, result])

  // ── Export PDF ───────────────────────────────────────
  const handleExportPDF = () => {
    if (!result) return
    const questionsHtml = questions.map((q, i) => {
      const answer = result.answers.find((a) => a.question_id === q.id)
      return `
        <div class="question-card ${answer?.is_correct ? 'correct' : 'incorrect'}">
          <div class="question-header">
            <span class="q-badge">${q.type === 'mcq' ? 'MCQ' : 'T/F'}</span>
            <p class="q-text">${i + 1}. ${q.question}</p>
          </div>
          ${q.options ? q.options.map((o) => `<div class="option">${o === q.correct_answer ? '✓ ' : ''}${o}</div>`).join('') : ''}
          <div class="result-text ${answer?.is_correct ? 'correct' : 'incorrect'}">
            ${answer?.is_correct ? '✓ Correct' : `✗ Incorrect — Correct Answer: ${q.correct_answer}`}
          </div>
        </div>`
    }).join('')
    const html = `
      <div class="stats">
        <div class="stat"><div class="stat-value">${result.score}%</div><div class="stat-label">Score</div></div>
        <div class="stat success"><div class="stat-value">${result.correct}</div><div class="stat-label">Correct</div></div>
        <div class="stat error"><div class="stat-value">${result.incorrect}</div><div class="stat-label">Incorrect</div></div>
      </div>${questionsHtml}`
    printAsPDF(`Quiz Results - ${materialTitle}`, html)
  }

  // ── Results view ─────────────────────────────────────
  if (isLoadingExistingQuiz && !isGenerating) {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card>
          <CardContent className="py-16">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-6 w-6 text-primary animate-spin" />
              <p className="text-sm text-muted-foreground text-center">
                Loading your latest quiz...
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  if (result && questions.length > 0) {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                {result.score >= 70
                  ? <CheckCircle2 className="h-6 w-6 text-green-500" />
                  : <XCircle className="h-6 w-6 text-red-500" />}
                Quiz Results
              </CardTitle>
              <CardDescription>
                {result.correct} of {result.total} correct
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleExportPDF}>
              <Printer className="mr-2 h-4 w-4" />Export PDF
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center p-4 bg-muted rounded-xl">
                <p className="text-3xl font-bold">{result.score}%</p>
                <p className="text-sm text-muted-foreground">Score</p>
              </div>
              <div className="text-center p-4 bg-green-500/10 rounded-xl">
                <p className="text-3xl font-bold text-green-600">{result.correct}</p>
                <p className="text-sm text-muted-foreground">Correct</p>
              </div>
              <div className="text-center p-4 bg-red-500/10 rounded-xl">
                <p className="text-3xl font-bold text-red-600">{result.incorrect}</p>
                <p className="text-sm text-muted-foreground">Incorrect</p>
              </div>
            </div>

            {/* Per-question breakdown */}
            <div className="space-y-3">
              {questions.map((question, index) => {
                const answer = result.answers.find((a) => a.question_id === question.id)
                return (
                  <div
                    key={question.id}
                    className={`p-4 rounded-xl border ${answer?.is_correct
                      ? 'border-green-500/30 bg-green-500/5'
                      : 'border-red-500/30 bg-red-500/5'
                      }`}
                  >
                    <div className="flex items-start gap-2">
                      {answer?.is_correct
                        ? <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                        : <XCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline" className="text-xs">
                            {question.type === 'mcq' ? 'MCQ' : 'True/False'}
                          </Badge>
                          <p className="font-medium text-foreground">
                            {index + 1}. {question.question}
                          </p>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Your answer: <span className="font-medium">{answer?.user_answer || 'Not answered'}</span>
                        </p>
                        {!answer?.is_correct && (
                          <p className="text-sm text-green-600 mt-0.5">
                            Correct: <span className="font-medium">{question.correct_answer}</span>
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
            <Button onClick={resetQuiz} className="w-full mt-6">Try Another Quiz</Button>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  // ── Active quiz view ─────────────────────────────────
  if (questions.length > 0) {
    const mcqQuestions = questions.filter((q) => q.type === 'mcq')
    const tfQuestions = questions.filter((q) => q.type === 'true_false')
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Quiz</CardTitle>
            <CardDescription className="flex items-center gap-2">
              <Badge variant="outline">{mcqQuestions.length} MCQ</Badge>
              <Badge variant="outline">{tfQuestions.length} True/False</Badge>
              <Badge variant="outline">{difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}</Badge>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {questions.map((question, index) => (
              <div key={question.id} className="space-y-3 pb-4 border-b last:border-0">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs flex-shrink-0">
                    {question.type === 'mcq' ? 'MCQ' : 'T/F'}
                  </Badge>
                  <p className="font-medium text-foreground">
                    {index + 1}. {question.question}
                  </p>
                </div>
                {question.type === 'mcq' ? (
                  <RadioGroup
                    value={answers[question.id] || ''}
                    onValueChange={(value) => setAnswers((prev) => ({ ...prev, [question.id]: value }))}
                    className="space-y-2 ml-2"
                  >
                    {question.options?.map((option, optIndex) => (
                      <div key={optIndex} className="flex items-center space-x-2">
                        <RadioGroupItem value={option} id={`${question.id}-${optIndex}`} />
                        <Label htmlFor={`${question.id}-${optIndex}`} className="cursor-pointer flex-1">
                          {option}
                        </Label>
                      </div>
                    ))}
                  </RadioGroup>
                ) : (
                  <RadioGroup
                    value={answers[question.id] || ''}
                    onValueChange={(value) => setAnswers((prev) => ({ ...prev, [question.id]: value }))}
                    className="flex gap-6 ml-2"
                  >
                    {['True', 'False'].map((opt) => (
                      <div key={opt} className="flex items-center space-x-2">
                        <RadioGroupItem value={opt} id={`${question.id}-${opt}`} />
                        <Label htmlFor={`${question.id}-${opt}`} className="cursor-pointer">{opt}</Label>
                      </div>
                    ))}
                  </RadioGroup>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
        <Button
          onClick={submitQuiz}
          disabled={Object.keys(answers).length !== questions.length}
          className="w-full"
          size="lg"
        >
          Submit Answers ({Object.keys(answers).length}/{questions.length} answered)
        </Button>
      </motion.div>
    )
  }

  // ── Config view ──────────────────────────────────────
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
      <Card>
        <CardHeader>
          <CardTitle>Generate Quiz</CardTitle>
          <CardDescription>Configure your quiz and test your knowledge</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">

          {/* Difficulty */}
          <div className="space-y-3">
            <Label className="text-base font-semibold">Difficulty</Label>
            <div className="flex gap-2">
              {(['easy', 'medium', 'hard'] as const).map((level) => (
                <Button
                  key={level}
                  variant={difficulty === level ? 'default' : 'outline'}
                  onClick={() => setDifficulty(level)}
                  className="flex-1"
                >
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </Button>
              ))}
            </div>
          </div>

          {/* Question Count Selectors */}
          <div className="space-y-6">
            <div className="flex flex-col gap-4">
              {/* MCQ Counter */}
              <div className="flex items-center justify-between p-4 rounded-2xl border border-border/50 bg-muted/30">
                <div className="flex flex-col">
                  <span className="text-base font-semibold">Multiple Choice</span>
                  <span className="text-sm text-muted-foreground">Range: 1-40</span>
                </div>
                <div className="flex items-center gap-4">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 rounded-full"
                    onClick={() => setMcqCount(Math.max(1, mcqCount - 1))}
                    disabled={mcqCount <= 1}
                  >
                    <Minus className="h-4 w-4" />
                  </Button>
                  <Input
                    type="number"
                    min={1}
                    max={40}
                    value={mcqCountInput}
                    onChange={(e) => {
                      const next = e.target.value
                      setMcqCountInput(next)
                      if (!next.trim()) return
                      const parsed = parseInt(next, 10)
                      if (!Number.isNaN(parsed)) {
                        setMcqCount(clampCount(parsed, 1, 40))
                      }
                    }}
                    onBlur={() => {
                      const parsed = parseInt(mcqCountInput, 10)
                      const clamped = clampCount(Number.isNaN(parsed) ? 1 : parsed, 1, 40)
                      setMcqCount(clamped)
                      setMcqCountInput(String(clamped))
                    }}
                    className="text-2xl font-bold w-16 text-center h-10 px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 rounded-full"
                    onClick={() => setMcqCount(Math.min(40, mcqCount + 1))}
                    disabled={mcqCount >= 40}
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* T/F Counter */}
              <div className="flex items-center justify-between p-4 rounded-2xl border border-border/50 bg-muted/30">
                <div className="flex flex-col">
                  <span className="text-base font-semibold">True / False</span>
                  <span className="text-sm text-muted-foreground">Range: 1-20</span>
                </div>
                <div className="flex items-center gap-4">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 rounded-full"
                    onClick={() => setTfCount(Math.max(1, tfCount - 1))}
                    disabled={tfCount <= 1}
                  >
                    <Minus className="h-4 w-4" />
                  </Button>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={tfCountInput}
                    onChange={(e) => {
                      const next = e.target.value
                      setTfCountInput(next)
                      if (!next.trim()) return
                      const parsed = parseInt(next, 10)
                      if (!Number.isNaN(parsed)) {
                        setTfCount(clampCount(parsed, 1, 20))
                      }
                    }}
                    onBlur={() => {
                      const parsed = parseInt(tfCountInput, 10)
                      const clamped = clampCount(Number.isNaN(parsed) ? 1 : parsed, 1, 20)
                      setTfCount(clamped)
                      setTfCountInput(String(clamped))
                    }}
                    className="text-2xl font-bold w-16 text-center h-10 px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 rounded-full"
                    onClick={() => setTfCount(Math.min(20, tfCount + 1))}
                    disabled={tfCount >= 20}
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Total Row */}
            <div className="pt-4 border-t border-border/50">
              <div className="flex items-center justify-between px-4 py-3 rounded-xl bg-primary/10 border border-primary/20">
                <span className="text-base font-semibold">Total Questions</span>
                <span className="text-xl font-bold text-primary">{mcqCount + tfCount}</span>
              </div>
            </div>
          </div>

          <Button onClick={generateQuiz} disabled={isGenerating} className="w-full" size="lg">
            {isGenerating ? (
              <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Generating Quiz...</>
            ) : (
              <><Brain className="mr-2 h-4 w-4" />Generate Quiz ({mcqCount + tfCount} questions)</>
            )}
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  )
}
