from typing import Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.state import app_state
from langchain.schema import Document
import datetime
from models.schemas import SourceState

async def ingest_webpage(url: str, description: Optional[str] = None) -> SourceState:
    """
    Uses vector store from global state. Process a source URL: extract content, and add it to the vector store, with embedding and splits.
    (Does not do any post-processing of the HTML)
    
    Args:
        url: The URL to process
        description: Optional description of the source
        
    Returns:
        A SourceState object with the status of the ingestion process.
    """
    
    try:
        # Load docs from the URL
        docs = await app_state.scraper.scrape_content(str(url), partial=True)
        
        # Print original document sizes
        for i, doc in enumerate(docs):
            print(f'DEBUG > Original doc {i+1}: {len(doc.page_content)} chars')
            print(f'DEBUG > First 100 chars: {doc.page_content[:100]}...')
        
        # Split docs into chunks for better retrieval
        # use RecursiveCharacterTextSplitter to optimize splitting of text
        # use tiktoken encoder to count tokens instead of chars
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, 
            chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs)
        
        # Debug the splits
        print(f'Split results: {len(doc_splits)} chunks. These are the first 3:')
        for i, split in enumerate(doc_splits[:3]):  # Show first 3 splits
            print(f'DEBUG > Split {i+1}:')
            print(f'  - Length: {len(split.page_content)} chars')
            print(f'  - Preview: {split.page_content[:100]}...')
            print(f'  - Metadata: {split.metadata}')

        # Add metadata to each doc_split
        ids = []
        for idx, doc in enumerate(doc_splits):
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['url'] = str(url)
            ids.append(f'{url}-SPLIT:{idx}') # specify and ID to allow upsert and prevent duplicates
        
        # Batch add documents to the vector store to reduce API calls to vector store and the embedding model
        batch_size = min(100, len(doc_splits))
        target_namespace = f'dev'

        print(f'Batching: {len(doc_splits)} splits into batches of {batch_size}')
        for i in range(0, len(doc_splits), batch_size):
            batch_docs = doc_splits[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            print(f'Processing batch {i//batch_size + 1}: {len(batch_docs)} documents')
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