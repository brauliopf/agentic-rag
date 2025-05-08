from fastapi import FastAPI, HTTPException, Body, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional, Literal
import uuid
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph.graph import MessagesState, StateGraph
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

# Data models
class SourceCreate(BaseModel):
    url: HttpUrl
    description: Optional[str] = None

class SourceResponse(BaseModel):
    id: str
    url: str
    description: Optional[str] = None
    status: Literal["pending", "processed", "failed"] = "pending"

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]


# Application state
class AppState:
    def __init__(self):
        self.sources: Dict[str, SourceResponse] = {}
        self.doc_splits = []
        self.retriever = None
        self.retriever_tool = None
        self.graph = None
        self.vectorstore = None
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        # Initialize language model
        self.llm = init_chat_model("openai:gpt-4.1", temperature=0)

# Initialize application state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    
    # context management: above runs on startup, below runs on teardown
    yield


app = FastAPI(lifespan=lifespan)


# Helper functions
def create_rag_graph(llm, retriever_tool):
    """Create the RAG workflow graph."""
    workflow = StateGraph(MessagesState)
    
    # Define nodes based on the functions in the notebook
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    
    # Define edges
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    
    return workflow.compile()


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state."""
    response = (
        app_state.llm.bind_tools([app_state.retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    from pydantic import BaseModel, Field
    
    class GradeDocuments(BaseModel):
        """Grade documents using a binary score for relevance check."""
        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )
    
    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    response = (
        app_state.llm
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )
    
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = app_state.llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


def generate_answer(state: MessagesState):
    """Generate an answer."""
    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \n"
        "Context: {context}"
    )
    
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = app_state.llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# Import necessary component after defining the graph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, END


# Create a utility function for source addition
def process_source(url, description=None):
    """Process a source URL and add it to the vector store"""
    source_id = str(uuid.uuid4())
    
    # Create source entry
    source_response = SourceResponse(
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
        return SourceResponse(
            id=source_id,
            url=str(url),
            status="failed"
        )


# Endpoints
@app.post("/sources", response_model=SourceResponse)
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


@app.get("/sources", response_model=List[SourceResponse])
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
    if not app_state.sources:
        raise HTTPException(status_code=400, detail="No sources have been added yet")
    
    # Run the query through the graph
    initial_state = {
        "messages": [
            {"role": "user", "content": query_request.query}
        ]
    }
    
    print(query_request.query)
    # Stream the response
    final_state = None
    for state in app_state.graph.stream(initial_state):
        # Keep track of the final state
        final_state = state
    
    # Extract the answer from the final state
    if final_state:
        last_message = final_state["messages"][-1]
        answer = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        # Extract sources (this is simplified - in practice, you would track which documents were used)
        source_urls = [s.url for s in app_state.sources.values() if s.status == "processed"]
        
        return QueryResponse(
            query=query_request.query,
            answer=answer,
            sources=source_urls
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to generate response")