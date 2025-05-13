from typing import Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.state import app_state
from langchain.schema import Document
import datetime
from models.schemas import SourceState

async def ingest_webpage(url: str, description: Optional[str] = None) -> SourceState:
    """
    Process a source URL: extract content, and add it to the vector store.
    (Does not do any post-processing of the HTML)
    
    Args:
        url: The URL to process
        description: Optional description of the source
        
    Returns:
        A SourceState object with the status of the ingestion process.
    """
    
    try:
        # Load docs from the URL
        docs = await app_state.scraper.scrape_content(str(url))

        # Playwright scraper returns a raw string with the HTML content
        # Convert to Document objects if needed
        if docs and isinstance(docs, str):
            docs = [Document(page_content=docs)]
        elif docs and isinstance(docs[0], str):
            docs = [Document(page_content=d) for d in docs]
        
        # Split docs into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs)

        # Add metadata to each doc_split
        ids = []
        for idx, doc in enumerate(doc_splits):
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['url'] = str(url)
            ids.append(f'{url}-SPLIT:{idx}')
        
        # Batch documents in smaller groups
        batch_size = 5  # Adjust based on your document sizes
        target_namespace = f'dev'
        for i in range(0, len(doc_splits), batch_size):
            batch_docs = doc_splits[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            app_state.vectorstore.add_documents(documents=batch_docs, ids=batch_ids, namespace=target_namespace)
        
        return SourceState(
            url=url,
            status="processed",
            scraped_at=datetime.datetime.now()
        )
    
    except Exception as e:
        print('Failed to add source:', e)
        return SourceState(
            url=url,
            status="failed"
        )