# Quiz Generator Routes

Generate and track quizzes (Multiple Choice Questions & True/False) from study materials or topics.

**Headers (All routes):**

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | Bearer Token | Yes | Supabase JWT token |

---

## `GET /api/quiz/list` — List Quizzes

Retrieve a list of generated quizzes. Optional filter by `material_id`.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `material_id` | string | No | Filter quizzes by study material ID |

**Response (200 OK):**

```json
[
  {
    "id": "quiz-uuid",
    "material_id": "material-uuid",
    "source_type": "pdf",
    "difficulty": "Medium",
    "mcq_count": 10,
    "tf_count": 5,
    "quiz_data": {
      "mcq": [...],
      "tf": [...]
    },
    "created_at": "2026-06-05T12:00:00Z"
  }
]
```

---

## `POST /api/quiz/generate` — Generate New Quiz

Generate a structured quiz from a topic, PDF material, or URL article. Daily limit check of 20 requests applies.

**Request Body (application/json):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `difficulty` | string | No | `"Medium"` | `"Easy"`, `"Medium"`, or `"Hard"` |
| `mcq_count` | int | No | `10` | Number of MCQ questions (1 to 20) |
| `tf_count` | int | No | `5` | Number of T/F questions (1 to 20) |
| `source_type` | string | Yes | `"web"` | `"web"`, `"topic"`, `"pdf"`, or `"url"` |
| `material_id` | string | No | `null` | Required if source_type is `"pdf"` or `"url"` |
| `topic` | string | No | `null` | Topic name (required for `"web"` or `"topic"` source_types if no material_id is supplied) |

**Response (200 OK):**

```json
{
  "quiz": {
    "mcq": [
      {
        "question": "What is the primary function of an operating system?",
        "options": ["A", "B", "C", "D"],
        "answer": "A",
        "explanation": "..."
      }
    ],
    "tf": [
      {
        "question": "An operating system directly runs system calls.",
        "answer": true,
        "explanation": "..."
      }
    ]
  },
  "quiz_id": "quiz-uuid"
}
```

---

## `POST /api/quiz/save-result` — Save Quiz Attempt

Save results (score, answers, etc.) for a completed quiz attempt.

**Request Body (application/json):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `quiz_id` | string | Yes | ID of the quiz |
| `result_data` | object | Yes | Key-value pairs detailing the user's answers and score |

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

## `GET /api/quiz/results/{quiz_id}` — Get Quiz Attempt Results

Retrieve saved attempts/results for a specific quiz.

**Response (200 OK):**

```json
[
  {
    "id": "result-uuid",
    "quiz_id": "quiz-uuid",
    "user_id": "user-uuid",
    "result_data": {
      "score": 80,
      "user_answers": [...]
    },
    "created_at": "2026-06-05T12:00:00Z"
  }
]
```
