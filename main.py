import streamlit as st
from graph_logic import create_graph

# 1. Page Configuration
st.set_page_config(page_title="AI Lead Assistant", page_icon="🏠", layout="wide")

st.title("🏠 Property Inquiry Lead Assistant")
st.markdown("""
    This assistant analyzes property issues, retrieves safety protocols, 
    and drafts professional, non-binding responses.
""")

# 2. Initialize the Graph
@st.cache_resource
def load_assistant():
    return create_graph()

assistant_app = load_assistant()

# 3. Sidebar for "Transparency" (Great for your Loom Video!)
st.sidebar.title("🔍 System Workflow Monitor")
st.sidebar.info("The documents AI considered while answering the user query.")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Explain your property issue here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process through LangGraph
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Prepare inputs for the graph
        inputs = {"user_query": prompt}
        
        # Stream the graph execution
        with st.status("Processing Workflow...", expanded=True) as status:
            for output in assistant_app.stream(inputs):
                for key, value in output.items():
                    # Update the sidebar based on which node is active
                    if key == "analyze_intent":
                        st.sidebar.subheader("1. Intent Analysis")
                        st.sidebar.write(value["intent_data"]["analysis"])
                    
                    elif key == "retrieve_knowledge":
                        st.sidebar.subheader("2. RAG Retrieval (FAISS)")
                        for doc in value["retrieved_context"]:
                            st.sidebar.write(f"✅ {doc}")
                    
                    elif key == "draft_response":
                        status.update(label="Response Generated!", state="complete", expanded=False)
                        full_response = value["draft_response"]
            
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# 5. Footer Instructions for the User
st.divider()
st.caption("Built with LangGraph + FAISS + Groq (openai/gpt-oss-20b). For demonstration.")