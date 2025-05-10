from typing import Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from models.schemas import SourceState


class AppState:
    """Manages the application state and resources."""
    
    def __init__(self):
        self.retriever = None # doc retrieval logic (engage with vector store)
        self.retriever_tool = None # make available to AI agents
        self.graph = None
        self.vectorstore = None
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        # Initialize language model
        self.llm = init_chat_model("openai:gpt-4.1", temperature=0)


# Initialize a global instance of the application state
app_state = AppState()