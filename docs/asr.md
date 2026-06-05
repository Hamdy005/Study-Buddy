# Audio Speech Recognition (ASR) Routes

## `POST /api/asr/transcribe` — Transcribe Audio File

Transcribe an uploaded audio recording using language-specific ASR models.

- **English model**: `nvidia/parakeet-tdt-0.6b-v2` (via NeMo)
- **Arabic model**: `IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53` (via Hugging Face Transformers)

The request is queued in a batch worker queue and returns the processed transcript.

**Headers:**

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | Bearer Token | Yes | Supabase JWT token |

**Request Body (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Audio recording file (e.g. `.webm`, `.wav`, `.ogg`) |
| `language` | string | Yes | Language code: `"en"` for English, `"ar"` for Arabic |

**Response (200 OK):**

```json
{
  "transcript": "hello world"
}
```

**Response Errors:**
- `400 Bad Request`: Invalid language selection.
- `500 Internal Server Error`: Transcription failed.
- `504 Gateway Timeout`: Audio transcription timed out (if it exceeds 60 seconds).

**Next.js Client Example:**

```ts
const formData = new FormData();
formData.append("audio", audioBlob, "recording.webm");
formData.append("language", "en");

const res = await fetch(`${BASE_URL}/api/asr/transcribe`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${token}`
  },
  body: formData,
});
const data = await res.json();
console.log(data.transcript);
```
