# ═══════════════════════ ASR Batch Worker Constants ════════════════════════

# Max audio files coalesced per batch window
ASR_BATCH_MAX = 6

# Seconds to collect concurrent jobs before firing inference
ASR_BATCH_WINDOW_S = 0.1

# Lightweight wav2vec2 warmup cadence (seconds)
WARMUP_INTERVAL_S = 45

# Parakeet (full model.transcribe) warmup every N lightweight cycles (~6 min at 45s each)
PARAKEET_WARMUP_EVERY = 8
