from pydantic import BaseModel, Field
from typing import List, Optional

class CultureScore(BaseModel):
    culture: str
    score: Optional[float] = Field(None, ge=1, le=5)  # None represents N/A for parsing errors
    reasoning: str

class WikipediaPage(BaseModel):
    title: str
    url: str
    rank: int
    score: float

class AnalysisResponse(BaseModel):
    scores: List[CultureScore]
    wikipedia_pages: List[WikipediaPage]
    matched_entity: str
    image_path: str
    session_id: Optional[str] = None  # Session ID for caching

class HealthResponse(BaseModel):
    status: str
    message: str

class PredefinedListsResponse(BaseModel):
    lists: List[str] 