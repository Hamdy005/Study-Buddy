# Authentication & User Profile Routes

Manage user account data, user profiles, and application settings (theme, display name, avatar). All routes require a valid Supabase JWT Bearer token.

**Headers (All routes):**

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | Bearer Token | Yes | Supabase JWT token |

---

## `GET /api/auth/profile` — Get User Profile

Fetches the logged-in user's profile details. If a database profile record does not yet exist (first-time login), this endpoint automatically pulls real user metadata from the Supabase Auth Admin API, upserts a new database record, and returns it.

**Response (200 OK):**

```json
{
  "status": "success",
  "user": {
    "id": "user-uuid",
    "display_name": "John Doe",
    "email": "john.doe@example.com",
    "avatar_url": "https://example.com/avatar.jpg",
    "theme": "dark",
    "daily_requests": 3,
    "last_request_date": "2026-06-05"
  }
}
```

---

## `PATCH /api/auth/profile` — Update User Profile

Updates user profile settings such as name, avatar URL, and theme.

**Request Body (application/json):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | No | New display name |
| `avatar_url` | string | No | URL of the profile picture |
| `theme` | string | No | UI Theme: `"light"` or `"dark"` |

**Response (200 OK):**

```json
{
  "status": "success",
  "user": {
    "id": "user-uuid",
    "display_name": "Updated Name",
    "email": "john.doe@example.com",
    "avatar_url": "https://example.com/new-avatar.jpg",
    "theme": "light",
    "daily_requests": 3,
    "last_request_date": "2026-06-05"
  }
}
```

---

## `DELETE /api/auth/me` — Delete Account Data

Deletes the user's account and associated metadata from the database.

**Response (200 OK):**

```json
{
  "status": "success",
  "message": "Account data deleted"
}
```
