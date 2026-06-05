# Summarizer Routes

Generate and fetch structured summaries for study materials.

**Headers (All routes):**

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | Bearer Token | Yes | Supabase JWT token |

---

## `POST /api/materials/summarize` — Generate Summary

Generate a structured summary for previously uploaded PDF files, URL articles, or Web topics. Daily limit check of 20 requests applies.

**Request Body (application/json):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `material_id` | string | Yes | The study material UUID |

**Response (200 OK):**

```json
{
  "summary": "# Summary Title\n\n- Key Point 1\n- Key Point 2",
  "time_taken": 4.52
}
```

---

## `GET /api/materials/{material_id}/summary` — Get Stored Summary

Fetch a previously generated summary for a material. Returns `null` if no summary has been generated yet.

**Response (200 OK):**

```json
{
  "summary": "# Summary Title\n\n- Key Point 1\n- Key Point 2",
  "time_taken": 4.52
}
```

or (if no summary generated yet):

```json
{
  "summary": null,
  "time_taken": 0
}
```
