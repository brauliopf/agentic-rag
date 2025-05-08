from fastapi import HTTPException
from models.schemas import QueryResponse
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
    if not app_state.sources:
        raise HTTPException(status_code=400, detail="No sources have been added yet")
    
    # Run the query through the graph
    initial_state = {
        "messages": [
            {"role": "user", "content": query_text}
        ]
    }
    
    # Run the query
    final_state = app_state.graph.invoke(initial_state)
    
    # Extract the answer from the final state
    if final_state and "messages" in final_state:
        # The last message in the messages array contains our answer
        last_message = final_state["messages"][-1]
        
        # Extract content from the message
        answer = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        # Extract sources (this is simplified - in practice, you would track which documents were used)
        source_urls = [s.url for s in app_state.sources.values() if s.status == "processed"]
        
        return QueryResponse(
            query=query_text,
            answer=answer,
            sources=source_urls
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to generate response") 