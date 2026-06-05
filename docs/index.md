# AI Tutor API — Documentation

Base URL: `http://localhost:8000`

Routes Overview:

- **Authentication & Profiles**: [auth.md](file:///home/mohamed-hamdy/1.My_Files/1.Data-Science/1.Projects/Study-Buddy/docs/auth.md)
- **Materials (PDFs, URLs, Topics)**: [materials.md](file:///home/mohamed-hamdy/1.My_Files/1.Data-Science/1.Projects/Study-Buddy/docs/materials.md)
- **Summarizer**: [summarizer.md](file:///home/mohamed-hamdy/1.My_Files/1.Data-Science/1.Projects/Study-Buddy/docs/summarizer.md)
- **Tutor (RAG & Chat Sessions)**: [rag.md](file:///home/mohamed-hamdy/1.My_Files/1.Data-Science/1.Projects/Study-Buddy/docs/rag.md)
- **Quiz Generator**: [quiz_generator.md](file:///home/mohamed-hamdy/1.My_Files/1.Data-Science/1.Projects/Study-Buddy/docs/quiz_generator.md)
- **Audio Speech Recognition (ASR)**: [asr.md](file:///home/mohamed-hamdy/1.My_Files/1.Data-Science/1.Projects/Study-Buddy/docs/asr.md)

Quick Start:

```bash
pip install -r src/requirements.txt
uvicorn src.main:app --reload --port 8000
```

All requests to endpoints (excluding public metadata check endpoints if any) require a Supabase JWT Bearer token in the `Authorization` header.
