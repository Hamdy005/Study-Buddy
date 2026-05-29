from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    material_id: str


class SummarizeResponse(BaseModel):
    summary: str
    time_taken: float
