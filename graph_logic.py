import os
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

# Define the State
class AgentState(TypedDict):
    user_query: str
    intent_data: dict
    retrieved_context: List[str]
    draft_response: str
    safety_check_passed: bool


llm = ChatGroq(
    temperature=0, model_name="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KY")
    )

vector_manager = VectorStoreManager()

# --- NODES ---

def analyze_intent_node(state: AgentState):
    """Identifies the problem and extracts key details."""
    prompt = f"""
    Analyze this customer inquiry: "{state['user_query']}"
    Extract: 
    1. Issue Type (e.g., Damp, Leak, Structural,out_of_scope_rule(if the query is not about property issues))
    2. Urgency (Low/Medium/High/out_of_scope_rule(if the query is not about property issues))
    3. Mentioned Details (Location, triggers,out_of_scope_rule(if the query is not about property issues))
    
    Return ONLY(strictly) a JSON-like format.

    """
    response = llm.invoke(prompt)
    return {"intent_data": {"analysis": response.content}}

def retrieve_knowledge_node(state: AgentState):
    """Fetches safe protocols from FAISS."""
    query = state['user_query']
    results = vector_manager.search(query)
    context = [res.page_content for res in results]
    return {"retrieved_context": context}

def draft_response_node(state: AgentState):
    """Drafts the final reply using RAG context and intent analysis."""
    context_str = "\n".join(state['retrieved_context'])
    prompt = f"""
    System: You are a professional Property Assistant. 
    Context from Safety Manual: {context_str}
    User Inquiry: {state['user_query']}
    
    Rules:
    1. Acknowledge the issue empathetically.
    2. Ask 2-3 specific clarifying questions if required (e.g., age of building).
    3. Provide safe next steps based on context.
    4. DO NOT make false promises or give cost estimates.
    5. Mention that a professional inspection is required.
    
    Draft a natural, human-like response:

    IMPORTANT:If the context "out_of_scope_rule" strictly reply "Please ask property related Questions"


    """
    response = llm.invoke(prompt)
    return {"draft_response": response.content}

# --- GRAPH CONSTRUCTION ---

def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("draft_response", draft_response_node)

    # Build Edges
    workflow.set_entry_point("analyze_intent")
    workflow.add_edge("analyze_intent", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "draft_response")
    workflow.add_edge("draft_response", END)

    return workflow.compile()

# Test the graph
if __name__ == "__main__":
    app = create_graph()
    inputs = {"user_query": "I have damp patches on my bedroom wall after heavy rain."}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"\n--- Node: {key} ---")
            print(value)