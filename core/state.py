from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from models.schemas import SourceState


class AppState:
    """Manages the application state and resources."""
    
    def __init__(self):
        self.sources: str = []
        self.retriever = None # doc retrieval logic (engage with vector store)
        self.retriever_tool = None # make available to AI agents
        self.graph = None
        self.vectorstore = None
        self.embeddings = None
        self.scraper = None
        self.llm = None


# Initialize a global instance of the application state
app_state = AppState()