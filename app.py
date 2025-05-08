import uuid
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.state import app_state
from models.schemas import (QueryRequest, QueryResponse, SourceCreate,
                            SourceState)
from rag.graph import create_rag_graph
from services.query import execute_query
from services.sources import process_source

load_dotenv()

# Use utility decorator to manage app resources before and after startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # context management 'YIELD': runs on startup
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
    
    yield
    # context management 'YIELD': here, runs below runs on teardown (empty fo now)


app = FastAPI(lifespan=lifespan)


# Create a utility function for source addition
def process_source(url, description=None):
    """Process a source URL and add it to the vector store"""
    source_id = str(uuid.uuid4())
    
    # Create source entry
    source_response = SourceState(
        id=source_id,
        url=str(url),
        description=description,
        status="pending"
    )
    app_state.sources[source_id] = source_response
    
    try:
        # Load docs from the URL
        docs = WebBaseLoader(str(url)).load()
        
        # Split docs into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        
        # Add to vectorstore
        app_state.vectorstore.add_documents(doc_splits)
        
        # Update source status
        app_state.sources[source_id].status = "processed"
        
        return source_response
    except Exception as e:
        app_state.sources[source_id].status = "failed"
        return SourceState(
            id=source_id,
            url=str(url),
            status="failed"
        )


# API Endpoints
@app.post("/sources", response_model=SourceState)
async def add_source(source: SourceCreate):
    """Add a new URL to the vector store"""
    result = process_source(source.url, source.description)
    
    if result.status == "failed":
        return JSONResponse(
            status_code=500,
            content={
                "id": result.id,
                "url": result.url,
                "status": "failed",
                "error": "Failed to process source"
            }
        )
    
    return result


@app.get("/sources", response_model=List[SourceState])
async def list_sources():
    """List all sources and their status"""
    return list(app_state.sources.values())


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str = Path(..., description="The ID of the source to delete")):
    """Delete a source and remove its embeddings from the vector store"""
    if source_id not in app_state.sources:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # For a real implementation, we would need a way to identify and remove 
    # specific documents from the vector store. Since InMemoryVectorStore doesn't 
    # provide a simple way to do this, in a production environment, you would:
    # 1. Store document/source ID mappings
    # 2. Use a vector store with deletion capabilities
    # 3. Rebuild the vector store excluding the deleted source
    
    # For this example, we'll just remove from our sources list
    del app_state.sources[source_id]
    
    return {"status": "deleted", "id": source_id}


@app.post("/query", response_model=QueryResponse)
async def query_sources(query_request: QueryRequest):
    """Query the sources and generate an answer"""
    return execute_query(query_request.query)