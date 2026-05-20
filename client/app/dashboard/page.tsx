'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plus,
  FileText,
  Link as LinkIcon,
  Upload,
  MoreHorizontal,
  Loader2,
  Clock,
  AlertCircle,
  CheckCircle2,
  FolderOpen,
  SquarePen,
  Pencil,
  Trash2,
  Search,
  X
} from 'lucide-react'
import { Logo } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'
import { UserDropdown } from '@/components/user-dropdown'
import { UsageStats } from '@/components/usage-stats'
import { useAuth } from '@/contexts/auth-context'
import { materialsAPI } from '@/lib/api'
import type { Material } from '@/lib/api'
import { cn } from '@/lib/utils'

const sourceTypeConfig: Record<string, { icon: React.ElementType; label: string; color: string }> = {
  pdf: { icon: FileText, label: 'PDF', color: 'bg-blue-500/10 text-blue-600' },
  url: { icon: LinkIcon, label: 'Article', color: 'bg-green-500/10 text-green-600' },
  topic: { icon: SquarePen, label: 'Topic', color: 'bg-purple-500/10 text-purple-600' },
}

const statusConfig: Record<string, { icon: React.ElementType; label: string; color: string; animate: boolean }> = {
  pending: { icon: Loader2, label: 'Uploading', color: 'bg-gray-500/10 text-gray-600', animate: true },
  processing: { icon: Loader2, label: 'Processing', color: 'bg-yellow-500/10 text-yellow-600', animate: true },
  ready: { icon: CheckCircle2, label: 'Ready', color: 'bg-green-500/10 text-green-600', animate: false },
  error: { icon: AlertCircle, label: 'Error', color: 'bg-red-500/10 text-red-600', animate: false },
  failed: { icon: AlertCircle, label: 'Failed', color: 'bg-red-500/10 text-red-600', animate: false },
}

const PENDING_UPLOAD_TIMEOUT_MS = 720000

export default function DashboardPage() {
  const router = useRouter()
  const { user, token } = useAuth()
  const [materials, setMaterials] = useState<Material[]>([])
  const [isUploadOpen, setIsUploadOpen] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [topicInput, setTopicInput] = useState('')
  const [isLoadingMaterials, setIsLoadingMaterials] = useState(true)
  const [materialsError, setMaterialsError] = useState(false)
  const [renameTarget, setRenameTarget] = useState<Material | null>(null)
  const [renameInput, setRenameInput] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)
  const [activeTab, setActiveTab] = useState('pdf')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [isSelectionMode, setIsSelectionMode] = useState(false)
  const [isBulkDeleting, setIsBulkDeleting] = useState(false)
  const [isBulkDeleteDialogOpen, setIsBulkDeleteDialogOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<string[] | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const loadingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const loadCounterRef = useRef(0)
  const pendingUploadTimeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const expiredPendingUploadsRef = useRef<Record<string, true>>({})

  // Track local title overrides so polling never wipes user renames
  const localTitlesRef = useRef<Record<string, string>>({})
  // Track temp → real ID mappings so we can clean up temp entries on poll
  const tempToRealRef = useRef<Record<string, string>>({})

  // ── Merge server data with local state ──────────────────────
  // Preserves: temp entries not yet resolved, local title overrides
  const mergeWithServer = (serverData: Material[]) => {
    const serverMap = new Map(serverData.map(m => [m.id, m]))

    // Apply local title overrides to server data
    for (const [id, title] of Object.entries(localTitlesRef.current)) {
      const mat = serverMap.get(id)
      if (mat) {
        serverMap.set(id, { ...mat, title })
      }
    }

    setMaterials(prev => {
      const prevMap = new Map(prev.map(m => [m.id, m]))

      // Keep later local statuses from regressing while the backend still reports pending
      for (const [id, material] of serverMap.entries()) {
        const previous = prevMap.get(id)
        if (!previous) continue
        if (material.status === 'pending' && (previous.status === 'processing' || previous.status === 'ready')) {
          serverMap.set(id, { ...material, status: previous.status })
        } else if (material.status === 'processing' && previous.status === 'ready') {
          serverMap.set(id, { ...material, status: 'ready' })
        }
      }

      // Keep temp entries that haven't been resolved yet
      const tempEntries = prev.filter(m =>
        m.id.startsWith('temp-') && !tempToRealRef.current[m.id]
      )
      // Apply local title overrides to temp entries too
      const updatedTemps = tempEntries.map(m =>
        localTitlesRef.current[m.id] ? { ...m, title: localTitlesRef.current[m.id] } : m
      )

      // Clean up resolved temp→real mappings from overrides
      for (const [tempId, realId] of Object.entries(tempToRealRef.current)) {
        // Transfer title override from temp to real if it exists
        if (localTitlesRef.current[tempId]) {
          localTitlesRef.current[realId] = localTitlesRef.current[tempId]
          const mat = serverMap.get(realId)
          if (mat) serverMap.set(realId, { ...mat, title: localTitlesRef.current[realId] })
          delete localTitlesRef.current[tempId]
        }
        delete tempToRealRef.current[tempId]
      }

      const merged = [...updatedTemps, ...Array.from(serverMap.values())]
      localStorage.setItem('cached_materials', JSON.stringify(merged))
      return merged
    })
  }

  // ── Polling ─────────────────────────────────────────────────
  const startPolling = () => {
    if (pollingRef.current) return
    pollingRef.current = setInterval(async () => {
      const data = await materialsAPI.list().catch(() => null)
      if (!data) return
      const sorted = data.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )
      mergeWithServer(sorted)
      const stillProcessing = sorted.some(
        (m) => m.status === 'processing' || m.status === 'pending'
      )
      if (!stillProcessing) {
        // Check if there are still temp entries waiting
        const hasTemps = Object.keys(tempToRealRef.current).length > 0
        if (!hasTemps) {
          clearInterval(pollingRef.current!)
          pollingRef.current = null
        }
      }
    }, 3000)
  }

  const clearPendingUploadTimeout = (tempId: string) => {
    const timeout = pendingUploadTimeoutsRef.current[tempId]
    if (timeout) {
      clearTimeout(timeout)
      delete pendingUploadTimeoutsRef.current[tempId]
    }
  }

  const removeTempMaterial = (tempId: string) => {
    setMaterials(prev => prev.filter(m => m.id !== tempId))
    setRenameTarget(prev => (prev?.id === tempId ? null : prev))
    delete localTitlesRef.current[tempId]
    delete tempToRealRef.current[tempId]
  }

  const startPendingUploadTimeout = (tempId: string) => {
    clearPendingUploadTimeout(tempId)
    pendingUploadTimeoutsRef.current[tempId] = setTimeout(() => {
      expiredPendingUploadsRef.current[tempId] = true
      removeTempMaterial(tempId)
      toast.error('Upload failed after taking too long. The material was removed from the dashboard.')
    }, PENDING_UPLOAD_TIMEOUT_MS)
  }

  useEffect(() => {
    const cached = localStorage.getItem('cached_materials')
    if (cached) {
      try {
        const parsed: Material[] = JSON.parse(cached)
        if (parsed.length > 0) {
          setMaterials(parsed)
          setIsLoadingMaterials(false)
        }
      } catch { }
    }
    loadMaterials()
  }, [token])

  const loadMaterials = async () => {
    const callId = ++loadCounterRef.current
    setIsLoadingMaterials(true)
    setMaterialsError(false)

    if (loadingTimeoutRef.current) clearTimeout(loadingTimeoutRef.current)
    loadingTimeoutRef.current = setTimeout(() => {
      if (loadCounterRef.current === callId) {
        setIsLoadingMaterials(false)
        setMaterials(prev => [...prev])
      }
    }, 10000)

    try {
      const data = await materialsAPI.list()
      if (loadCounterRef.current !== callId) return []
      const sorted = data.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )
      mergeWithServer(sorted)
      return sorted
    } catch (err: unknown) {
      if (loadCounterRef.current !== callId) return []
      console.error('Failed to load materials:', err)
      const cached = localStorage.getItem('cached_materials')
      if (!cached) {
        setMaterialsError(true)
        setMaterials([])
      }
      return []
    } finally {
      if (loadCounterRef.current === callId) {
        setIsLoadingMaterials(false)
        if (loadingTimeoutRef.current) clearTimeout(loadingTimeoutRef.current)
        loadingTimeoutRef.current = null
      }
    }
  }

  // ── Search ──────────────────────────────────────────────────
  useEffect(() => {
    let active = true

    if (!searchQuery.trim()) {
      setSearchResults(null)
      return
    }

    setSearchResults([])

    const timer = setTimeout(async () => {
      try {
        const res = await materialsAPI.search(searchQuery.trim())
        if (active) {
          setSearchResults(res.results.map(r => r.material_id))
        }
      } catch {
        if (active) {
          setSearchResults([])
        }
      }
    }, 300)

    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [searchQuery])

  const normalizedSearchQuery = searchQuery.trim().toLowerCase()
  const searchResultIds = new Set(searchResults ?? [])

  const displayMaterials = normalizedSearchQuery
    ? materials.filter(m =>
        searchResultIds.has(m.id) ||
        m.title.toLowerCase().includes(normalizedSearchQuery)
      )
    : materials

  // ── Upload Handlers ─────────────────────────────────────────
  const handleFileUpload = (file: File) => {
    if (!file.type.includes('pdf')) {
      toast.error('Please upload a PDF file')
      return
    }

    const tempId = 'temp-' + Date.now()
    const tempMaterial: Material = {
      id: tempId,
      title: file.name.replace('.pdf', ''),
      source_type: 'pdf',
      status: 'pending',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    // Close dialog, add temp entry, open rename — all instant
    setIsUploadOpen(false)
    setMaterials(prev => [tempMaterial, ...prev])
    setRenameTarget(tempMaterial)
    setRenameInput(tempMaterial.title)
    startPendingUploadTimeout(tempId)

    // Upload in background — completely non-blocking
    materialsAPI.uploadPDF(file)
      .then((result) => {
        clearPendingUploadTimeout(tempId)
        if (expiredPendingUploadsRef.current[tempId]) {
          delete expiredPendingUploadsRef.current[tempId]
          materialsAPI.delete(result.material_id).catch(() => { })
          return
        }
        const newMat: Material = {
          id: result.material_id,
          title: localTitlesRef.current[tempId] || result.title || file.name.replace('.pdf', ''),
          source_type: 'pdf',
          status: 'processing',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
        tempToRealRef.current[tempId] = result.material_id
        // Transfer any local title override from temp to real
        if (localTitlesRef.current[tempId]) {
          localTitlesRef.current[result.material_id] = localTitlesRef.current[tempId]
          // Also rename on server since user renamed before upload finished
          materialsAPI.rename(result.material_id, localTitlesRef.current[tempId]).catch(() => { })
        }
        setMaterials(prev => prev.map(m => {
          if (m.id === tempId) {
            const existingMaterial = prev.find(p => p.id === result.material_id && !p.id.startsWith('temp-'))
            return {
              ...newMat,
              status: existingMaterial?.status || newMat.status
            }
          }
          return m
        }))
        setRenameTarget(prev => (prev?.id === tempId ? { ...newMat, status: prev.status === 'ready' ? 'ready' : 'processing' } : prev))
        toast.success('PDF uploaded! Processing in background...')
        startPolling()
      })
      .catch((err) => {
        clearPendingUploadTimeout(tempId)
        if (expiredPendingUploadsRef.current[tempId]) {
          delete expiredPendingUploadsRef.current[tempId]
          return
        }
        const msg = err instanceof Error ? err.message : 'Upload failed'
        toast.error(msg)
        removeTempMaterial(tempId)
      })
  }

  const handleURLSubmit = () => {
    if (!urlInput.trim()) {
      toast.error('Please enter a URL')
      return
    }

    const tempId = 'temp-' + Date.now()
    let hostname = 'Article'
    try { hostname = new URL(urlInput.trim()).hostname } catch { /* */ }

    const tempMaterial: Material = {
      id: tempId,
      title: 'Article from ' + hostname,
      source_type: 'url',
      status: 'pending',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    const url = urlInput.trim()
    setIsUploadOpen(false)
    setUrlInput('')
    setMaterials(prev => [tempMaterial, ...prev])
    setRenameTarget(tempMaterial)
    setRenameInput(tempMaterial.title)
    startPendingUploadTimeout(tempId)

    materialsAPI.scrapeURL(url)
      .then((result) => {
        clearPendingUploadTimeout(tempId)
        if (expiredPendingUploadsRef.current[tempId]) {
          delete expiredPendingUploadsRef.current[tempId]
          materialsAPI.delete(result.material_id).catch(() => { })
          return
        }
        const newMat: Material = {
          id: result.material_id,
          title: localTitlesRef.current[tempId] || result.title || 'New Article',
          source_type: 'url',
          status: 'processing',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
        tempToRealRef.current[tempId] = result.material_id
        if (localTitlesRef.current[tempId]) {
          localTitlesRef.current[result.material_id] = localTitlesRef.current[tempId]
          materialsAPI.rename(result.material_id, localTitlesRef.current[tempId]).catch(() => { })
        }
        setMaterials(prev => prev.map(m => {
          if (m.id === tempId) {
            const existingMaterial = prev.find(p => p.id === result.material_id && !p.id.startsWith('temp-'))
            return {
              ...newMat,
              status: existingMaterial?.status || newMat.status
            }
          }
          return m
        }))
        setRenameTarget(prev => (prev?.id === tempId ? { ...newMat, status: prev.status === 'ready' ? 'ready' : 'processing' } : prev))
        toast.success('URL added! Processing in background...')
        startPolling()
      })
      .catch((err) => {
        clearPendingUploadTimeout(tempId)
        if (expiredPendingUploadsRef.current[tempId]) {
          delete expiredPendingUploadsRef.current[tempId]
          return
        }
        const msg = err instanceof Error ? err.message : 'Failed to add URL'
        toast.error(msg)
        removeTempMaterial(tempId)
      })
  }

  const handleTopicSubmit = async () => {
    if (!topicInput.trim()) {
      toast.error('Please enter a topic')
      return
    }

    const normalizedTitle = topicInput.trim().toLowerCase()
    const duplicateTitle = materials.some((m) => m.title.trim().toLowerCase() === normalizedTitle)
    if (duplicateTitle) {
      toast.error('A material with this title already exists')
      return
    }

    setIsUploadOpen(false)
    try {
      await materialsAPI.addTopic(topicInput.trim())
      toast.success('Topic added successfully!')
      setTopicInput('')
      loadMaterials()
    } catch (err: unknown) {
      toast.error('Failed to add topic')
    }
  }

  // ── Rename ──────────────────────────────────────────────────
  const handleRename = () => {
    if (!renameTarget || !renameInput.trim()) return
    const targetId = renameTarget.id
    const newTitle = renameInput.trim()
    const normalizedTitle = newTitle.toLowerCase()
    const hasConflict = materials.some(
      (m) => m.id !== targetId && m.title.trim().toLowerCase() === normalizedTitle
    )
    if (hasConflict) {
      toast.error('Title already exists. Please choose a unique name.')
      return
    }

    // Store in local overrides so polling can never wipe it
    localTitlesRef.current[targetId] = newTitle

    // Close dialog and update UI instantly
    setRenameTarget(null)
    setRenameInput('')
    setMaterials(prev => {
      const updated = prev.map(m => m.id === targetId ? { ...m, title: newTitle } : m)
      localStorage.setItem('cached_materials', JSON.stringify(updated))
      return updated
    })
    toast.success('Material renamed!')

    // Fire API in background — only for real (non-temp) IDs
    if (!targetId.startsWith('temp-')) {
      materialsAPI.rename(targetId, newTitle).then(() => {
        startPolling()
      }).catch((err) => {
        const msg = err instanceof Error ? err.message : 'Failed to save rename — please try again'
        toast.error(msg)
      })
    }
    // For temp IDs, the rename will be synced when upload completes (see upload handlers)
  }


  const handleBulkDelete = async () => {
    if (selectedIds.length === 0) return
    setIsBulkDeleting(true)
    setIsBulkDeleteDialogOpen(false)
    try {
      const realIds = selectedIds.filter(id => !id.startsWith('temp-'))
      if (realIds.length > 0) {
        await materialsAPI.bulkDelete(realIds)
      }

      const updated = materials.filter((m) => !selectedIds.includes(m.id))
      setMaterials(updated)
      localStorage.setItem('cached_materials', JSON.stringify(updated))
      // Clean up local title overrides for deleted IDs
      for (const id of selectedIds) {
        delete localTitlesRef.current[id]
      }
      toast.success(`${selectedIds.length} materials deleted`)
      exitSelectionMode()
    } catch (err: unknown) {
      toast.error('Failed to delete some materials')
      loadMaterials()
    } finally {
      setIsBulkDeleting(false)
    }
  }

  const toggleSelection = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation()
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    )
  }

  const enterSelectionMode = (id: string) => {
    setIsSelectionMode(true)
    setSelectedIds([id])
  }

  const exitSelectionMode = () => {
    setIsSelectionMode(false)
    setSelectedIds([])
  }

  const toggleSelectAll = () => {
    if (selectedIds.length === displayMaterials.length) {
      setSelectedIds([])
    } else {
      setSelectedIds(displayMaterials.map(m => m.id))
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  }

  const isCurrentRenameTargetDuplicate = renameTarget ? materials.some(
    (m) => m.id !== renameTarget.id && m.title.trim().toLowerCase() === renameTarget.title.trim().toLowerCase()
  ) : false;

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
              <span className="text-sm text-muted-foreground hidden sm:block">
                {user?.name || 'User'}
              </span>
              <UserDropdown />
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex-1">
              <h1 className="text-2xl sm:text-3xl font-bold text-foreground">My Materials</h1>
              <p className="text-muted-foreground mt-1">
                {searchResults !== null
                  ? `${displayMaterials.length} of ${materials.length} found`
                  : selectedIds.length > 0
                    ? `${selectedIds.length} of ${materials.length} selected`
                    : 'Upload and manage your learning materials'}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <AnimatePresence>
                {isSelectionMode && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.6 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.6 }}
                  >
                    <button
                      onClick={toggleSelectAll}
                      title={selectedIds.length === displayMaterials.length ? 'Deselect All' : 'Select All'}
                      className={cn(
                        "w-9 h-9 rounded-full border-2 flex items-center justify-center transition-all duration-200 shadow-sm",
                        selectedIds.length === displayMaterials.length
                          ? "bg-primary border-primary text-primary-foreground"
                          : "bg-card border-border text-muted-foreground hover:border-primary/60 hover:text-foreground"
                      )}
                    >
                      <CheckCircle2 className="w-4 h-4" />
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>

              <Dialog open={isUploadOpen} onOpenChange={setIsUploadOpen}>
                <DialogTrigger asChild>
                  <Button size="lg">
                    <Plus className="mr-2 h-4 w-4" />
                    Add Material
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-lg">
                  <DialogHeader>
                    <DialogTitle>Add New Material</DialogTitle>
                    <DialogDescription>
                      Upload a PDF, add an article URL, or enter a custom topic
                    </DialogDescription>
                  </DialogHeader>

                  <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-4">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="pdf" className="gap-2 hover:bg-primary/5 transition-colors">
                        <FileText className="h-4 w-4" />
                        <span className="hidden sm:inline">PDF</span>
                      </TabsTrigger>
                      <TabsTrigger value="url" className="gap-2 hover:bg-primary/5 transition-colors">
                        <LinkIcon className="h-4 w-4" />
                        <span className="hidden sm:inline">URL</span>
                      </TabsTrigger>
                      <TabsTrigger value="topic" className="gap-2 hover:bg-primary/5 transition-colors">
                        <SquarePen className="h-4 w-4" />
                        <span className="hidden sm:inline">Topic</span>
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="pdf" className="mt-4">
                      <input
                        type="file"
                        ref={fileInputRef}
                        accept=".pdf"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0]
                          if (file) handleFileUpload(file)
                        }}
                      />
                      <div
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-border rounded-xl p-8 text-center cursor-pointer hover:border-primary/50 hover:bg-muted/30 transition-colors"
                      >
                        <Upload className="h-10 w-10 mx-auto text-muted-foreground mb-4" />
                        <p className="text-foreground font-medium">Click to upload or drag and drop</p>
                        <p className="text-sm text-muted-foreground mt-1">PDF files up to 10MB</p>
                      </div>
                    </TabsContent>

                    <TabsContent value="url" className="mt-4 space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="url">Article URL</Label>
                        <Input
                          id="url"
                          placeholder="https://example.com/article"
                          value={urlInput}
                          onChange={(e) => setUrlInput(e.target.value)}
                        />
                      </div>
                      <Button
                        onClick={handleURLSubmit}
                        disabled={!urlInput.trim()}
                        className="w-full"
                      >
                        <LinkIcon className="mr-2 h-4 w-4" />
                        Add Article
                      </Button>
                    </TabsContent>

                    <TabsContent value="topic" className="mt-4 space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="topic">Custom Topic</Label>
                        <Input
                          id="topic"
                          placeholder="e.g. Neural Networks, Redis Caching, CI/CD Pipelines"
                          value={topicInput}
                          onChange={(e) => setTopicInput(e.target.value)}
                        />
                        <p className="text-xs text-muted-foreground">
                          Study Buddy will use AI to generate summaries, quizzes and answer questions about this topic.
                        </p>
                      </div>
                      <Button
                        onClick={handleTopicSubmit}
                        disabled={!topicInput.trim()}
                        className="w-full"
                      >
                        <SquarePen className="mr-2 h-4 w-4" />
                        Add Topic
                      </Button>
                    </TabsContent>
                  </Tabs>

                </DialogContent>
              </Dialog>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Search materials..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 pr-9"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        <AnimatePresence mode="popLayout">
          {isLoadingMaterials ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : materialsError ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-16"
            >
              <div className="w-20 h-20 mx-auto bg-destructive/10 rounded-2xl flex items-center justify-center mb-6">
                <AlertCircle className="w-10 h-10 text-destructive" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Failed to load materials</h3>
              <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
                Could not connect to the server. Your materials are stored securely in the database.
              </p>
              <Button onClick={loadMaterials} variant="outline">
                <Loader2 className="mr-2 h-4 w-4" />
                Retry
              </Button>
            </motion.div>
          ) : displayMaterials.length === 0 && searchResults !== null ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-16"
            >
              <div className="w-20 h-20 mx-auto bg-muted rounded-2xl flex items-center justify-center mb-6">
                <Search className="w-10 h-10 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">No results found</h3>
              <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
                No materials match "{searchQuery}". Try a different search term.
              </p>
              <Button onClick={() => setSearchQuery('')} variant="outline">
                <X className="mr-2 h-4 w-4" />
                Clear search
              </Button>
            </motion.div>
          ) : materials.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-16"
            >
              <div className="w-20 h-20 mx-auto bg-muted rounded-2xl flex items-center justify-center mb-6">
                <FolderOpen className="w-10 h-10 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">No materials yet</h3>
              <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
                Upload your first learning material to get started with AI-powered studying
              </p>
              <Button onClick={() => setIsUploadOpen(true)}>
                <Plus className="mr-2 h-4 w-4" />
                Add Your First Material
              </Button>
            </motion.div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {displayMaterials.map((material, index) => {
                const sourceType = sourceTypeConfig[material.source_type] || sourceTypeConfig.pdf
                const status = statusConfig[material.status] || statusConfig.error
                const SourceIcon = sourceType.icon
                const StatusIcon = status.icon

                const isTemp = material.id.startsWith('temp-')

                return (
                  <motion.div
                    key={material.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <Card
                      className={cn(
                        "group transition-all duration-300 relative overflow-hidden",
                        isTemp ? "opacity-70 cursor-not-allowed border-dashed" : "cursor-pointer hover:shadow-lg hover:border-primary/30",
                        selectedIds.includes(material.id) && "border-primary ring-1 ring-primary/30"
                      )}
                      onClick={() => {
                        if (isTemp) return
                        if (isSelectionMode) {
                          toggleSelection(material.id)
                        } else {
                          router.push(`/material/${material.id}`)
                        }
                      }}
                    >
                      {/* Small circle bubble at top-right — only in selection mode */}
                      {!isTemp && isSelectionMode && (
                        <div
                          className="absolute top-3 right-3 z-10"
                          onClick={(e) => toggleSelection(material.id, e)}
                        >
                          <div className={cn(
                            "w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200 shadow-sm",
                            selectedIds.includes(material.id)
                              ? "bg-primary border-primary text-primary-foreground scale-110"
                              : "bg-card border-border text-muted-foreground hover:border-primary/60"
                          )}>
                            {selectedIds.includes(material.id) && (
                              <CheckCircle2 className="w-3.5 h-3.5" />
                            )}
                          </div>
                        </div>
                      )}

                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className={cn("p-2.5 rounded-xl", sourceType.color)}>
                            <SourceIcon className="w-5 h-5" />
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary" className={status.color}>
                              <StatusIcon
                                className={`w-3 h-3 mr-1 ${status.animate ? 'animate-spin' : ''}`}
                              />
                              {status.label}
                            </Badge>
                            {!isTemp && (
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                                  <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                                  {material.source_type !== 'topic' && (
                                    <DropdownMenuItem
                                      className="cursor-pointer"
                                      onClick={() => {
                                        setRenameTarget(material)
                                        setRenameInput(material.title)
                                      }}
                                    >
                                      <Pencil className="mr-2 h-4 w-4" />
                                      Rename
                                    </DropdownMenuItem>
                                  )}
                                  <DropdownMenuItem
                                    className="cursor-pointer text-destructive focus:text-destructive"
                                    onClick={() => enterSelectionMode(material.id)}
                                  >
                                    <Trash2 className="mr-2 h-4 w-4" />
                                    Delete
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            )}
                          </div>
                        </div>

                        <h3 className="font-semibold text-foreground mb-2 line-clamp-2 group-hover:text-primary transition-colors">
                          {material.title}
                        </h3>

                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Clock className="w-3.5 h-3.5" />
                          <span>{formatDate(material.created_at)}</span>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          )}
        </AnimatePresence>

        {/* Bulk Actions Bar */}
        <AnimatePresence>
          {selectedIds.length > 0 && (
            <motion.div
              initial={{ y: 100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 100, opacity: 0 }}
              className="fixed bottom-8 left-1/2 -translate-x-1/2 z-[60] w-full max-w-md px-4"
            >
              <div className="bg-card text-foreground border border-border rounded-2xl p-4 shadow-2xl flex items-center justify-between gap-4 backdrop-blur-md">
                <div className="flex items-center gap-3 pl-1">
                  <div className="w-8 h-8 bg-primary/10 text-primary rounded-lg flex items-center justify-center font-bold text-sm">
                    {selectedIds.length}
                  </div>
                  <span className="font-medium text-foreground">Selected</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground hover:text-foreground"
                    onClick={exitSelectionMode}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => setIsBulkDeleteDialogOpen(true)}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete {selectedIds.length}
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Robust Deletion Blocking Modal */}
        <Dialog open={isBulkDeleting}>
          <DialogContent className="sm:max-w-md text-center p-12 pointer-events-none" hideCloseButton>
            <div className="flex flex-col items-center">
              <div className="w-20 h-20 bg-destructive/10 rounded-3xl flex items-center justify-center mb-6">
                <Loader2 className="w-10 h-10 text-destructive animate-spin" />
              </div>
              <DialogTitle className="text-2xl font-bold mb-2">Deleting Materials</DialogTitle>
              <DialogDescription className="text-base">
                Please wait while we securely remove your data.
                <br /> This action cannot be undone.
              </DialogDescription>
              <div className="mt-8 flex items-center gap-2 text-sm text-muted-foreground bg-muted px-4 py-2 rounded-full">
                <div className="flex items-center gap-1">
                  {selectedIds.length} items being processed
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Bulk Delete Confirmation */}
        <AlertDialog open={isBulkDeleteDialogOpen} onOpenChange={setIsBulkDeleteDialogOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete {selectedIds.length} Materials?</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete all selected materials? This will permanently remove all associated summaries, quizzes, and chat histories.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                onClick={handleBulkDelete}
              >
                Delete {selectedIds.length} items
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </main>

      <Dialog 
        open={!!renameTarget} 
        onOpenChange={(open) => { 
          if (!open) {
            setRenameTarget(null) 
          }
        }}
      >
        <DialogContent 
          className="sm:max-w-md"
        >
          <DialogHeader>
            <DialogTitle>Rename Material</DialogTitle>
            <DialogDescription>Enter a new name for this material.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <Input
              value={renameInput}
              onChange={(e) => setRenameInput(e.target.value)}
              placeholder="Material name"
              onKeyDown={(e) => { if (e.key === 'Enter') handleRename() }}
              autoFocus
            />
            {renameTarget && renameInput.trim() && materials.some(
              (m) => m.id !== renameTarget.id && m.title.trim().toLowerCase() === renameInput.trim().toLowerCase()
            ) && (
                <p className="text-sm text-destructive">A material with this title already exists.</p>
              )}
            <div className="flex justify-end gap-2">
              <Button 
                variant="outline" 
                onClick={() => setRenameTarget(null)}
              >Cancel</Button>
              <Button
                onClick={handleRename}
                disabled={!renameInput.trim() || (renameTarget ? materials.some(
                  (m) => m.id !== renameTarget.id && m.title.trim().toLowerCase() === renameInput.trim().toLowerCase()
                ) : false)}
              >
                Rename
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
