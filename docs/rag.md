# Tutor & RAG Routes

Interactive study buddy bot that uses context from uploaded files (PDFs, URLs), Wikipedia, and DuckDuckGo web search to guide learning. Supports persistent chat sessions.

**Headers (All routes):**

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | Bearer Token | Yes | Supabase JWT token |

---

## `POST /api/tutor/ask` — Ask Study Buddy

Submit a question to the AI Tutor. The tutor retrieves matching excerpts from user documents, pulls Web search snippets, handles safety/NSFW filtering, and generates an answer. 

**Request Body (application/json):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | — | Question or prompt from the student |
| `source_type` | string | No | `"web"` | `"web"`, `"topic"`, `"pdf"`, or `"url"` |
| `material_id` | string | No | `null` | Required if source_type is `"pdf"` or `"url"` |
| `session_id` | string | No | `null` | Chat session UUID to persist conversation history |
| `memory_id` | string | No | `null` | Legacy session identifier |

**Response (200 OK):**

```json
{
  "answer": "The Application Binary Interface (ABI)...",
  "source": "PDF (embeddings vector)",
  "time_taken": 2.14,
  "memory_id": "session-uuid-or-memory-id"
}
```

---

## `GET /api/tutor/sessions` — List Chat Sessions

List all chat sessions associated with a specific material.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `material_id` | string | Yes | The study material UUID |

**Response (200 OK):**

```json
[
  {
    "id": "session-uuid",
    "material_id": "material-uuid",
    "user_id": "user-uuid",
    "title": "Introduction to ABI",
    "created_at": "2026-06-05T12:00:00Z"
  }
]
```

---

## `POST /api/tutor/sessions` — Create Chat Session

Create a new chat session for a study material.

**Request Body (application/json):**

```json
{
  "material_id": "material-uuid",
  "title": "New Chat Session"
}
```

**Response (200 OK):**

```json
{
  "id": "session-uuid",
  "material_id": "material-uuid",
  "user_id": "user-uuid",
  "title": "New Chat Session",
  "created_at": "2026-06-05T12:00:00Z"
}
```

---

## `GET /api/tutor/sessions/{session_id}/messages` — Get Session Messages

Loads the message history for a session.

**Response (200 OK):**

```json
[
  {
    "id": "message-uuid",
    "session_id": "session-uuid",
    "role": "user",
    "content": "What is an ABI?",
    "created_at": "2026-06-05T12:00:00Z"
  },
  {
    "id": "message-uuid-2",
    "session_id": "session-uuid",
    "role": "assistant",
    "content": "An Application Binary Interface (ABI)...",
    "created_at": "2026-06-05T12:00:02Z"
  }
]
```

---

## `PATCH /api/tutor/sessions/{session_id}` — Rename Session

Renames an existing chat session.

**Request Body (application/json):**

```json
{
  "title": "Updated Session Name"
}
```

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

## `DELETE /api/tutor/sessions/{session_id}` — Delete Session

Deletes a chat session and all its messages.

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

## `POST /api/tutor/sessions/{session_id}/extract-title` — Auto-Rename Session

Automatically generates a 3-5 word title based on the user's initial message/query and renames the session.

**Request Body (application/json):**

```json
{
  "query": "What is an ABI and how does it invoke system calls?"
}
```

**Response (200 OK):**

```json
{
  "status": "ok",
  "title": "ABI & System Calls"
}
```

---

## `POST /api/tutor/chat/save` — Save Chat History (Legacy)

Saves chat messages associated with a material (legacy API).

**Request Body (application/json):**

```json
{
  "material_id": "material-uuid",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ]
}
```

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

## `GET /api/tutor/chat/{material_id}` — Load Chat History (Legacy)

Loads chat messages associated with a material (legacy API).

**Response (200 OK):**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ]
}
```
