import uuid
from typing import Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from models.schemas import SourceState
from core.state import app_state
from services.scraper import webscraper



def ingest_webpage(url: str, description: Optional[str] = None):
    """
    Process a source URL, extract content, and add it to the vector store.
    
    Args:
        url: The URL to process
        description: Optional description of the source
        
    Returns:
        A SourceState object representing the processed source
    """
    
    try:
        # Load docs from the URL
        docs = WebBaseLoader(str(url)).load()
        
        # Split docs into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)

        # Add metadata to each doc_split
        ids = []
        for idx, doc in enumerate(doc_splits):
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['url'] = url
            ids.append(f'{url}-SPLIT:{idx}')
        
        # Add to vectorstore
        target_namespace = f'dev'
        app_state.vectorstore.add_documents(documents=doc_splits, ids=ids, namespace=target_namespace)
        
        return True
    
    except Exception as e:
        print('Failed to add source')
        return False

async def process_source_play(url: str, description: Optional[str] = None) -> SourceState:
    """
    Asynchronous version of process_source.
    
    Args:
        url: The URL to process
        description: Optional description of the source
        
    Returns:
        A SourceState object representing the processed source
    """
    source_id = str(uuid.uuid4())
    
    # Create source entry
    source_response = SourceState(
        id=source_id,
        url=str(url),
        status="pending",
        description=description,
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
        target_namespace = "some_specific_namespace"
        app_state.vectorstore.add_documents(doc_splits, namespace=target_namespace)

        # Update source status
        app_state.sources[source_id].status = "processed"
        
        return source_response
    except Exception as e:
        app_state.sources[source_id].status = "failed"
        return SourceState(
            id=source_id,
            url=str(url),
            status="failed",
            description=description
        ) 