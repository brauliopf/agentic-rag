from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.state import app_state
from rag.graph import create_rag_graph
from rag.sources import ingest_webpage
from rag.vectorstore import get_vector_store
from langchain.tools.retriever import create_retriever_tool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifecycle events.Handles initialization on startup and cleanup on shutdown."""

    # Initialize Pinecone vector store
    app_state.vectorstore = get_vector_store()
    app_state.retriever = app_state.vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.5},
    )
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
            # ("https://lilianweng.github.io/posts/2024-07-07-hallucination/", "page 2"),
            # ("https://lilianweng.github.io/posts/2024-04-12-diffusion-video/", "page 3")
        ]
        
        for url, description in default_sources:
            ingest_webpage(url)
            
        print('Added default sources')
    except Exception as e:
        print(f"Error adding default sources: {e}")
    
    # Above code runs on startup
    yield
    # Below code runs on shutdown (currently empty)
