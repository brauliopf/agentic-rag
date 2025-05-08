from typing import Dict
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from models.schemas import SourceState


class AppState:
    """Manages the application state and resources."""
    
    def __init__(self):
        self.sources: Dict[str, SourceState] = {}
        self.doc_splits = []
        self.retriever = None
        self.retriever_tool = None
        self.graph = None
        self.vectorstore = None
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        # Initialize language model
        self.llm = init_chat_model("openai:gpt-4.1", temperature=0)


# Initialize a global instance of the application state
app_state = AppState()