from fastapi import HTTPException
from models.schemas import QueryResponse, GraphState
from core.state import app_state


def execute_query(query_text):
    """Process a user query and generate an answer using the RAG graph.
    
    Args:
        query_text: The user's query text
        
    Returns:
        QueryResponse object with the query, answer, and source URLs
        
    Raises:
        HTTPException: If no sources have been added or if the query fails
    """
    # if not app_state.sources:
    #     raise HTTPException(status_code=400, detail="No sources have been added yet")
    
    # Run the query through the graph
    initial_state = GraphState(
        question=query_text
    )
    
    # Run the query
    final_state = app_state.graph.invoke(initial_state)
    
    # Extract the answer and sources from the final state
    if final_state and "answer" in final_state:
        agent_answer = final_state["answer"]
        
        # Extract content from the message
        answer = agent_answer.content if hasattr(agent_answer, "content") else str(agent_answer)
        
        # Extract unique source URLs from context documents' metadata
        sources = []
        if final_state.get("context"):
            for doc in final_state["context"]:
                if doc.metadata and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
        
        return QueryResponse(
            query=query_text,
            answer=answer,
            sources=sources if sources else ['No sources found']
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to generate response") 