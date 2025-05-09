from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.state import app_state
from rag.graph import create_rag_graph
from rag.sources import process_source
from rag.vectorstore import initialize_vectorstore

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.
    
    Handles initialization on startup and cleanup on shutdown.
    """
    # Initialize vector store, retriever, and retriever tool
    app_state.vectorstore, app_state.retriever, app_state.retriever_tool = initialize_vectorstore(
        embeddings=app_state.embeddings
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
