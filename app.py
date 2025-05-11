import uuid
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from core.state import app_state
from core.lifespan import lifespan
from models.schemas import (QueryRequest, QueryResponse, SourceCreate, SourceState)
from core.query import execute_query
from rag.sources import ingest_webpage

load_dotenv()

app = FastAPI(lifespan=lifespan)


# API Endpoints
@app.get("/sources", response_model=List[SourceState])
async def list_sources():
    """List all sources and their status"""
    return app_state.sources


@app.post("/sources", response_model=SourceState)
async def add_source(source: SourceCreate):
    """Add a new URL to the vector store"""
    result = await ingest_webpage(str(source.url), source.description)
    # Optionally, store str(source.url) in app_state.sources if needed
    app_state.sources.append(str(source.url))
    if result.status == "failed":
        return JSONResponse(
            status_code=500,
            content={
                "id": getattr(result, "id", None),
                "url": str(result.url),
                "status": "failed",
                "error": "Failed to process source"
            }
        )
    return result

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