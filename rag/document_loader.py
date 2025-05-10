from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)
from langchain_community.document_loaders.playwright import PlaywrightURLLoader
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional
import os
import asyncio
from playwright.async_api import async_playwright

def load_documents(source_dir: str) -> List[Document]:
    """
    Load documents from a directory with support for different file types.
    """
    # Set up loaders for different file types
    loaders = {
        ".pdf": DirectoryLoader(source_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        ".txt": DirectoryLoader(source_dir, glob="**/*.txt", loader_cls=TextLoader)
    }
    
    documents = []
    
    # Load each file type
    for file_type, loader in loaders.items():
        if any(f.endswith(file_type) for f in os.listdir(source_dir)):
            try:
                documents.extend(loader.load())
                print(f"Loaded {file_type} documents from {source_dir}")
            except Exception as e:
                print(f"Error loading {file_type} documents: {e}")
    
    return documents

async def scrape_webpage_content(url: str, wait_for: Optional[str] = None) -> str:
    """
    Advanced web scraper that can handle dynamic content.
    
    Args:
        url: URL to scrape
        wait_for: Optional CSS selector to wait for before scraping
        
    Returns:
        Extracted text content
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()
        
        # Set viewport size
        await page.set_viewport_size({"width": 1280, "height": 1080})
        
        try:
            # Navigation with timeout
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait for specific element if requested
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=10000)
            else:
                # Default: wait a bit for dynamic content
                await page.wait_for_timeout(2000)
            
            # Extract content
            content = await page.content()
            
            # Extract text content
            text_content = await page.evaluate("""() => {
                // Remove script and style elements
                document.querySelectorAll('script, style, nav, footer, header, aside').forEach(el => el.remove());
                
                // Get main content
                const main = document.querySelector('main') || document.querySelector('article') || document.body;
                
                // Remove unnecessary whitespace and normalize
                return main.innerText
                    .replace(/\\s+/g, ' ')
                    .replace(/\\n+/g, '\\n')
                    .trim();
            }""")
            
            # Get metadata
            title = await page.title()
            description = await page.evaluate("""() => {
                const meta = document.querySelector('meta[name="description"]');
                return meta ? meta.getAttribute('content') : '';
            }""")
            
            return {
                "text": text_content,
                "metadata": {
                    "source": url,
                    "title": title,
                    "description": description
                }
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {"text": "", "metadata": {"source": url, "error": str(e)}}
        finally:
            await browser.close()

def load_webpages(urls: List[str], wait_for_selectors: Optional[Dict[str, str]] = None) -> List[Document]:
    """
    Load webpages using Playwright and convert to Documents.
    
    Args:
        urls: List of URLs to scrape
        wait_for_selectors: Optional dictionary mapping URLs to CSS selectors to wait for
        
    Returns:
        List of Document objects
    """
    if wait_for_selectors is None:
        wait_for_selectors = {}
    
    async def process_all_urls():
        tasks = []
        for url in urls:
            wait_for = wait_for_selectors.get(url)
            tasks.append(scrape_webpage_content(url, wait_for))
        return await asyncio.gather(*tasks)
    
    # Run the async scraping
    results = asyncio.run(process_all_urls())
    
    # Convert to Documents
    documents = []
    for result in results:
        if result["text"]:
            documents.append(Document(
                page_content=result["text"],
                metadata=result["metadata"]
            ))
    
    print(f"Loaded {len(documents)} web pages")
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks for better processing.
    
    Best practices:
    - chunk_size: 1000-1500 characters works well for most applications
    - chunk_overlap: ~20% of chunk_size helps maintain context between chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    return split_docs

def prepare_documents_for_vectorstore(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Prepare documents for ingestion into Pinecone.
    Adds unique IDs and formats metadata properly.
    """
    prepared_docs = []
    
    for i, doc in enumerate(documents):
        prepared_docs.append({
            "id": f"doc_{i}",
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    return prepared_docs