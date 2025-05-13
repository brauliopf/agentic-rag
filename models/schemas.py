from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

class GraphState(TypedDict):
    question: str
    context: List[Document] = None
    answer: str = None

class SourceCreate(BaseModel):
    url: HttpUrl
    description: Optional[str]


class SourceState(BaseModel):
    url: HttpUrl
    status: Literal["pending", "processed", "failed"] = "pending"
    text: Optional[str] = None
    scraped_at: Optional[datetime] = None


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