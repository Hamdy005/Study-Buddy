# Study Buddy Email Templates (Supabase Auth)

Custom, high-deliverability email templates with the official Study Buddy logo, clean typography, anti-phishing compliance, and no raw URL fallback boxes.

---

## Where to apply in Supabase Dashboard

Go to: **Authentication** &rarr; **Emails** &rarr; **Email Templates**

| Template File | Supabase Email Template Tab | Subject Line |
|---|---|---|
| [`confirm-signup.html`](./confirm-signup.html) | **Confirm signup** | `Verify your email for Study Buddy` |
| [`reset-password.html`](./reset-password.html) | **Reset Password** | `Reset your password for Study Buddy` |
| [`magic-link.html`](./magic-link.html) | **Magic Link** | `Your magic sign-in link for Study Buddy` |
| [`change-email.html`](./change-email.html) | **Change Email Address** | `Confirm your new email address for Study Buddy` |
| [`invite-user.html`](./invite-user.html) | **Invite user** | `You've been invited to Study Buddy` |

---

## Key Features
- **Official Study Buddy Logo**: Uses `https://studybuddyai.dev/apple-icon.png` (high-res PNG supported across 100% of desktop, mobile, and webmail clients).
- **Clean Action Button**: Prominent `#4B6EE5` brand button.
- **No URL Leakage**: Removed the raw `Trouble clicking the button?` box to prevent spam filter heuristic penalties.
- **Anti-Spam Footer**: Legitimate reason statement and domain attribution to maintain a high domain reputation score.
