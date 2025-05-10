from typing import Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.state import app_state
from langchain.schema import Document

async def ingest_webpage(url: str, description: Optional[str] = None):
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
        docs = await app_state.scraper.scrape_content(str(url))
        
        # Playwright scraper returns a raw string with the HTML content
        # Convert to Document objects if needed
        if docs and isinstance(docs, str):
            docs = [Document(page_content=docs)]
        elif docs and isinstance(docs[0], str):
            docs = [Document(page_content=d) for d in docs]
        
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
        
        # Batch documents in smaller groups
        batch_size = 20  # Adjust based on your document sizes
        target_namespace = f'dev'
        for i in range(0, len(doc_splits), batch_size):
            batch_docs = doc_splits[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            app_state.vectorstore.add_documents(documents=batch_docs, ids=batch_ids, namespace=target_namespace)
        
        return True
    
    except Exception as e:
        print('Failed to add source:', e)
        return False