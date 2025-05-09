import uuid
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from core.state import app_state
from core.lifespan import lifespan
from models.schemas import (QueryRequest, QueryResponse, SourceCreate, SourceState)
from services.query import execute_query
from rag.sources import ingest_webpage, process_source_play

load_dotenv()

app = FastAPI(lifespan=lifespan)


# API Endpoints
@app.get("/sources", response_model=List[SourceState])
async def list_sources():
    """List all sources and their status"""
    return list(app_state.sources.values())


# @app.get("/sources/{source_id}", response_model=SourceState)
# async def get_source(source_id: str = Path(..., description="The ID of the source to retrieve")):
#     """Get metadata and content for a specific source"""
#     if source_id not in app_state.sources:
#         raise HTTPException(status_code=404, detail="Source not found")
    
#     return app_state.sources[source_id]


@app.post("/sources", response_model=SourceState)
def add_source(source: SourceCreate):
    """Add a new URL to the vector store"""
    # result = process_source(source.url, source.description)
    result = ingest_webpage(source.url, source.description)
    
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


@app.post("/sources_play", response_model=SourceState)
async def add_source_play(source: SourceCreate):
    """Add a new URL to the vector store"""
    # result = process_source(source.url, source.description)
    result = await process_source_play(source.url, source.description)
    
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