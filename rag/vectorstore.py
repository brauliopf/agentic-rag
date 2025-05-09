from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_core.embeddings import Embeddings
from typing import List, Optional
from langchain_core.documents import Document

def initialize_vectorstore(embeddings: Embeddings, documents: Optional[List[Document]] = None) -> tuple:
    """
    Initialize a vector store with the provided embeddings and documents.
    
    Args:
        embeddings: The embeddings model to use for document vectorization
        documents: Optional list of documents to pre-populate the vector store
        
    Returns:
        A tuple containing (vectorstore, retriever, retriever_tool)
    """
    # Create the vector store with initial documents if provided
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents or [], 
        embedding=embeddings
    )
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()
    
    # Create a tool that can be used by agents to access the retriever
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_sources",
        "Search and return information from the loaded sources.",
    )
    
    return vectorstore, retriever, retriever_tool