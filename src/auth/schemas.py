from pydantic import BaseModel
from typing import Optional

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    theme: Optional[str] = None
