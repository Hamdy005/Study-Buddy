# Materials Routes

Manage study materials (PDFs, Web URLs, and custom Topics) for learning, summarization, and chat.

**Headers (All routes):**

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | Bearer Token | Yes | Supabase JWT token |

---

## `GET /api/materials` — List Materials

Lists all materials created by the logged-in user.

**Response (200 OK):**

```json
[
  {
    "id": "material-uuid",
    "user_id": "user-uuid",
    "title": "Introduction to OS",
    "source_type": "pdf",
    "url": null,
    "status": "ready",
    "error_message": null,
    "created_at": "2026-06-05T12:00:00Z"
  }
]
```

---

## `GET /api/materials/{material_id}` — Get Material Details

Retrieves details of a specific material by its ID.

**Response (200 OK):**

```json
{
  "id": "material-uuid",
  "user_id": "user-uuid",
  "title": "Introduction to OS",
  "source_type": "pdf",
  "url": null,
  "status": "ready",
  "error_message": null,
  "created_at": "2026-06-05T12:00:00Z"
}
```

---

## `POST /api/materials/upload-pdf` — Upload PDF

Upload a PDF file to create a new material. Starts a background task to extract text, create chunks, and generate/store vector embeddings.

**Request Body (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | The PDF file (max size: 10MB) |

**Response (200 OK):**

```json
{
  "status": "processing_started",
  "material_id": "material-uuid",
  "title": "filename.pdf"
}
```

---

## `POST /api/materials/scrape-url` — Scrape URL

Create a new material by scraping content from a webpage URL. Starts a background task to scrape, chunk, and embed content.

**Request Body (application/json):**

```json
{
  "url": "https://example.com/article"
}
```

**Response (200 OK):**

```json
{
  "status": "processing_started",
  "material_id": "material-uuid",
  "title": "https://example.com/article"
}
```

---

## `POST /api/materials/topic` — Create Custom Topic

Create a plain topic container (no file upload or URL scraping) for chat/quizzes. Performs local safety validation (NSFW/political/religious terms) before creation.

**Request Body (application/json):**

```json
{
  "topic": "Operating Systems Concepts"
}
```

**Response (200 OK):**

```json
{
  "material_id": "material-uuid",
  "title": "Operating Systems Concepts"
}
```

---

## `PATCH /api/materials/{material_id}` — Rename Material

Rename an existing material (PDF or URL). Renaming custom topic containers is disabled.

**Request Body (application/json):**

```json
{
  "title": "New Material Title"
}
```

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

## `POST /api/materials/search` — Search Materials

Search for materials belonging to the user matching a text query in the title.

**Request Body (application/json):**

```json
{
  "q": "Operating Systems"
}
```

**Response (200 OK):**

```json
{
  "results": [
    {
      "id": "material-uuid",
      "title": "Operating Systems Concepts"
    }
  ]
}
```

---

## `DELETE /api/materials/{material_id}` — Delete Material

Deletes a study material, its chunks, summary, generated quizzes, and embeddings.

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

## `POST /api/materials/bulk-delete` — Bulk Delete Materials

Deletes multiple materials in parallel.

**Request Body (application/json):**

```json
{
  "material_ids": ["uuid-1", "uuid-2"]
}
```

**Response (200 OK):**

```json
{
  "status": "ok"
}
```
