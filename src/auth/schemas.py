from pydantic import BaseModel, Field
from typing import Optional

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    theme: Optional[str] = None
    current_password: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)


class EmailRateLimitRequest(BaseModel):
    email: str
    action: str  # 'email_verification', 'forgot_password', or 'change_password_confirmation'



