from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Literal


class SourceCreate(BaseModel):
    url: HttpUrl
    description: Optional[str] = None


class SourceState(BaseModel):
    id: str
    url: str
    status: Literal["pending", "processed", "failed"] = "pending"
    content: Optional[str] = None
    description: Optional[str] = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )