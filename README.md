<p>
  <img src="client/public/icon.svg" width="50" valign="middle">
  <strong>
  <span style="font-size:50px; vertical-align:middle;">
   &nbsp;Study Buddy
  </span>
</strong>
</p>

--- 
Study Buddy is a Next.js AI tutor powered by FastAPI and Supabase. Upload PDFs and URLs to generate structured summaries, multi-session chat, and custom quizzes. Custom topics use web search through Wikipedia, ArXiv, and DuckDuckGo.

---

## 🏗️ Architecture Overview

### 🌐 Frontend — Next.js

The UI is built with **Next.js** and deployed on **Vercel**, served under a custom domain. It supports **light and dark mode**: the default follows the **system preference**, and the user can override it with a manual toggle.

### ⚡ Backend — FastAPI

The AI engine runs on a **FastAPI** backend, handling embeddings, summarization, chat, and quiz generation.

### 🗄️ Database — Supabase

**Supabase** serves as the central database and vector store. All user data is persisted:

---

## ✨ Frontend Features

You can upload a **PDF**, provide a **URL**, or enter a **custom topic** — the system generates embeddings (for PDFs and URLs) or uses web search (for custom topics) to power all the features below.

### 📝 Summary Generator

After generation, the summary is displayed as a **well-structured document** with **sub-header boxes** — each subtopic is stored in its own card for easy reading. Summaries can also be **exported**.

### 💬 Chat (Multi-Session)

- Users can create **multiple chat sessions**, each with its own context.
- Every chat session is **stored in the database** so it can be retrieved or revisited later.
- Chats are **exportable** in a pdf.

### 🧪 Quiz Generator

- Supports **True/False** and **MCQ** questions.
- Users can choose **any custom number of questions**.
- Quizzes are rendered in a **clean, good-looking UI** and can be **exported**.

---

## 🔐 Authentication

**Google OAuth** is used for all authentication — no custom passwords, usernames, or sign-up forms.

---

## 🧠 AI & Backend Details

### Embedding Pipeline

- Uses a **multilingual embedder** for cross-language support.
- When a material is uploaded, the system **processes it and generates embeddings asynchronously**, allowing **concurrent embedding of multiple materials** in parallel rather than sequentially.
- A **warm-up cycle** keeps the embedder active — ML models become idle without a forward pass for extended periods, so periodic warm-up prevents cold starts.

### Summary & Quiz Generation

- Uses **OpenRouter's API key** to generate summaries and quizzes based on material embeddings.
- If a **custom topic** is uploaded (instead of a PDF or URL), the topic text itself is used directly for generation.

### Chatbot

- Uses **Gemma 4 31B** (`google/gemma-4-31b-it`) as primary model via Google Gemini API, with automatic fallback to **Gemma 4 26B** (`google/gemma-4-26b-it`) on rate limit.
- For **PDF and URL materials**, the chatbot replies using the stored **embeddings**.
- Each chat session gets an **auto-generated title** based on the first user message.
- For **custom topic** materials (no PDF/URL), the chatbot falls back to **web search tools** — **Wikipedia, ArXiv, and DuckDuckGo Search** — to retrieve relevant information.

### Rate Limits

- **Summary & Quiz generation** — 20 requests per day per user (uses Google Gemini API).
- **Chatbot** — Powered by Google Gemini API (`google/gemma-4-31b-it` & `google/gemma-4-26b-it`).

---

## 📸 Screenshots

### 🔑 Google Sign-In
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

---

## 🔧 Tools & Technologies

| Component | Technology |
|-----------|------------|
| Frontend | Next.js (Vercel + custom domain) |
| Backend | FastAPI (Python) |
| Database & Vector Store | Supabase |
| Summary & Quiz Model | `gemini-3.1-flash-lite` via Google Gemini API |
| Chatbot Model | `google/gemma-4-31b-it` (fallback: `google/gemma-4-26b-it`) via Gemini API |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Auth | Google OAuth |
| Web Search | Wikipedia, ArXiv, DuckDuckGo |
