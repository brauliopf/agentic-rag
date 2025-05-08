from contextlib import asynccontextmanager
from fastapi import FastAPI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from core.state import app_state
from rag.graph import create_rag_graph
from services.sources import process_source

# Use utility decorator to manage app resources before and after startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.
    
    Handles initialization on startup and cleanup on shutdown.
    """
    # Initialize in-memory vector store
    app_state.vectorstore = InMemoryVectorStore.from_documents(
        documents=[], embedding=app_state.embeddings
    )
    app_state.retriever = app_state.vectorstore.as_retriever()
    app_state.retriever_tool = create_retriever_tool(
        app_state.retriever,
        "retrieve_sources",
        "Search and return information from the loaded sources.",
    )
    
    # Setup the RAG graph
    app_state.graph = create_rag_graph(app_state.llm, app_state.retriever_tool)
    
    # Initialize with default sources
    try:
        # Add default sources
        default_sources = [
            ("https://lilianweng.github.io/posts/2024-11-28-reward-hacking/", "page 1"),
            ("https://lilianweng.github.io/posts/2024-07-07-hallucination/", "page 2"),
            ("https://lilianweng.github.io/posts/2024-04-12-diffusion-video/", "page 3")
        ]
        
        for url, description in default_sources:
            process_source(url, description)
            
        print('Added default sources')
    except Exception as e:
        print(f"Error adding default sources: {e}")
    
    # Above code runs on startup
    yield
    # Below code runs on shutdown (currently empty)
