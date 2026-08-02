<p>
  <img src="client/public/icon.svg" width="50" valign="middle">
  <strong>
  <span style="font-size:50px; vertical-align:middle;">
   &nbsp;Study Buddy
  </span>
</strong>
</p>

--- 
Study Buddy is an intelligent Next.js AI study assistant powered by FastAPI, Supabase, Cloudinary, and Resend. Upload PDFs, audio files, or web URLs to generate structured summaries, multi-session contextual chat, and custom quizzes. For custom study topics, Study Buddy leverages web search engines (Wikipedia & DuckDuckGo).

---

## 🏗️ Architecture Overview

### 🌐 Frontend — Next.js
The UI is built with **Next.js 15 (App Router)**. It supports **light and dark mode** (auto-detecting system preference with manual toggle override).

### ⚡ Backend — FastAPI
The AI engine runs on **FastAPI**, handling embedding generation, document processing, ASR audio transcription, multi-session RAG chat, summarization, and quiz generation.

### 🗄️ Database & Vector Store — Supabase
**Supabase** serves as the central PostgreSQL database and vector store (`pgvector`). All user accounts, materials, chunks, chat histories, summaries, and quizzes are securely stored.

### ☁️ Cloud Storage — Cloudinary
Uploaded PDF files and media assets are hosted on **Cloudinary**, providing scalable file storage and high-speed CDN delivery for PDF preview rendering.

### 📧 Email Service — Resend
Custom authentication workflows (email verification links, password reset tokens) are powered by **Resend** for reliable transactional email delivery.

---

## ✨ Features

### 🔐 Authentication & Account Management
- **Google OAuth**: One-click login via Google Identity Services.
- **Custom Email / Password**: Sign up and sign in with email verification powered by **Resend**.
- **Password Reset**: Secure email-based password recovery flow with token validation.
- **Account Deletion**: Full self-service account removal with cascade cleanup of user data, uploaded materials, and chat histories.

### 📁 Material Uploads & Data Processing
- **PDF Uploads**: Processed, text-sanitized, chunked, and stored in Supabase with original file assets hosted on **Cloudinary**.
- **Audio Files & ASR**: Speech-to-Text transcription powered by **NeMo Toolkit ASR** (English & Arabic models).
- **Web URLs & Custom Topics**: Live content retrieval via Wikipedia & DuckDuckGo search APIs.

### 📝 Summary Generator
- Generates structured, high-clarity document summaries organized into visual subtopic cards.
- Powered by **Gemini 3.5 Flash Lite** (fallback to **Gemini 3.1 Flash Lite**).
- Exportable to PDF.

### 💬 Multi-Session RAG Chatbot
- Powered by Mistral AI's **Ministral 8B** (`ministral-8b-latest`).
- Multi-session conversation management with persistent chat history.
- Context-aware RAG querying over uploaded PDF/URL chunks.
- Auto-generated chat titles based on session context.
- Exportable chat transcripts.

### 🧪 Quiz Generator
- Generates **Multiple Choice (MCQ)** and **True/False** questions.
- Powered by **Gemini 3.5 Flash Lite** with a 12,000 output token limit (fallback to **Gemini 3.1 Flash Lite**).
- Customizable difficulty levels and question counts.
- Interactive quiz interface with immediate scoring and PDF export.

---

## 🧠 AI Models & Infrastructure

| Component | Provider / Technology | Details |
|-----------|------------------------|---------|
| **Frontend** | Next.js 15 + React | Vercel Deployment |
| **Backend** | FastAPI (Python 3.12/3.10) | Async API Architecture |
| **Database & Vector Store** | Supabase | PostgreSQL + `pgvector` |
| **Cloud Asset Storage** | Cloudinary | CDN-hosted PDF & media uploads |
| **Transactional Email** | Resend | Signup verification & password reset |
| **RAG Chatbot Model** | Mistral AI (`ministral-8b-latest`) | High-performance 8B edge LLM |
| **Summary Model** | Google Gemini (`gemini-3.5-flash-lite`) | Fallback: `gemini-3.1-flash-lite` |
| **Quiz Model** | Google Gemini (`gemini-3.5-flash-lite`) | Fallback: `gemini-3.1-flash-lite` |
| **Text Embeddings** | HuggingFace / SentenceTransformers | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Speech Recognition (ASR)** | NVIDIA NeMo Toolkit | English & Arabic ASR models |
| **Authentication** | Google OAuth & Custom Auth | Google Identity Services + Resend Email |
| **Web Search Tools** | Wikipedia & DuckDuckGo APIs | Fallback search for topic materials |

---

## 📸 Screenshots

### 🔑 Google Sign-In & Email Authentication
<p align="center">
  <img src="https://github.com/user-attachments/assets/df7a002f-eabe-41e8-b0d3-9a51c7ba5b94" alt="Google Sign-In" width="800"/>
</p>

### 🏠 Main Dashboard
<p align="center">
  <img src="https://github.com/user-attachments/assets/46e98ff1-1933-4688-8285-e5b30018023c" alt="Main Dashboard" width="800"/>
</p>

### 📝 Generated Summary
<p align="center">
  <img src="https://github.com/user-attachments/assets/478edde7-d692-40a7-bcdd-b7a63f839504" alt="Generated Summary" width="800"/>
  <br/>
  <br/>
  <img src="https://github.com/user-attachments/assets/d0468650-6f27-4838-97f9-20ffd8fc75aa" alt="Generated Summary" width="800"/>
</p>

### 💬 Chat Session
<p align="center">
  <img src="https://github.com/user-attachments/assets/01ed1e77-60bf-4bc5-a300-91ff66469c4d" alt="Chat Session with multiple sessions" width="800"/>
</p>

### 🧪 Quiz
<p align="center">
  <img src="https://github.com/user-attachments/assets/e23a8de6-f55b-4fbe-ada6-c3ef93202374" alt="Generate Quiz" width="800"/>
  <br/>
  <br/>
  <img src="https://github.com/user-attachments/assets/3e009295-2f2d-4d53-81a4-bbd6c2700134" alt="Generated Quiz" width="800"/>
  <br/>
  <br/>
  <img src="https://github.com/user-attachments/assets/8c623980-665b-49bb-9c3a-b0f2fe741008" alt="Quiz Results" width="800"/>
</p>

### 📄 Export PDF (Quiz)
<p align="center">
  <img src="https://github.com/user-attachments/assets/dcc2659f-da33-4e7a-9c36-6d31b0ef3200" alt="Export Quiz as PDF" width="800"/>
</p>
