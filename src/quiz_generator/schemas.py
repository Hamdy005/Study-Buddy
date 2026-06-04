from pydantic import BaseModel
from typing import Optional

class QuizRequest(BaseModel):
    difficulty: str = "Medium"
    mcq_count: int = 10
    tf_count: int = 5
    source_type: str = "web"
    material_id: Optional[str] = None
    topic: Optional[str] = None


class QuizResponse(BaseModel):
    quiz: dict
    quiz_id: str


class SaveQuizResultRequest(BaseModel):
    quiz_id: str
    result_data: dict
