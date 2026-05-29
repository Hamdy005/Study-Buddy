from pydantic import BaseModel

class URLInput(BaseModel):
    url: str


class RenameMaterialRequest(BaseModel):
    title: str


class BulkDeleteRequest(BaseModel):
    material_ids: list[str]


class TopicRequest(BaseModel):
    topic: str


class SearchRequest(BaseModel):
    q: str
