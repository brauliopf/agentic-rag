import uuid
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.schemas import SourceState
from core.state import app_state


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