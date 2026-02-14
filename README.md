# Lead-Response-Assistant---Property-Inquiry-Lead-Assistant
run on terminal "streamlit run main.py" for the prototype

##Project Overview: Lead Response Assistant

a. How the System Works:
The system utilizes a Stateful Agentic Workflow powered by LangGraph, Groq (Llama-3), and FAISS. It avoids linear prompting in favor of a structured graph logic:
-Intent Categorization Node: Analyzes the user's raw input to identify the specific property issue (e.g., dampness, leaks) and urgency level.
-RAG Retrieval Node: Queries a local FAISS Vector Store containing verified safety protocols and company SOPs to find "Grounded Truth" relevant to the issue.
-State Management: A TypedDict state carries data (user query, intent, retrieved context) across nodes, ensuring the LLM has all necessary facts before drafting.
-Response Generation Node: Synthesizes the retrieved technical context with the user's query to produce a professional, empathetic reply.

b. Ensuring Accuracy and Reliability
To meet the requirement of a "Safe" and "Client-Ready" assistant, the following mechanisms are implemented:
-Retrieval-Augmented Generation (RAG): By grounding the LLM in a vector database, the system is forced to use provided protocols rather than relying on internal      parametric memory, which prevents hallucinations regarding repair costs.
-Deterministic Reasoning: Setting the model Temperature to 0 ensures consistent, non-creative, and professional outputs across different runs.
-Negative Constraint Enforcement: The system prompt explicitly forbids specific high-risk behaviors, such as providing cost estimates or guaranteeing fixes without   an inspection.
-Modular Node Logic: Separating "Thinking" (Analysis) from "Writing" (Generation) reduces cognitive load on the LLM, leading to higher logical precision.

c. Known Limitations
-Text-Only Modality: The assistant cannot visually inspect photos of the damage. It relies entirely on the user's verbal description, which may be inaccurate or      incomplete.
-Vector Semantic Collisions: In cases of extremely vague input, the FAISS similarity search may retrieve broadly related documents that aren't perfectly specific     to the niche sub-issue.
-Static Knowledge Base: The current FAISS index is built from a static source; it does not yet support real-time updates from a live technical database or CRM.

d. Future Improvements
-Cross-Encoder Reranking: Adding a reranking stage after retrieval to mathematically verify the semantic relevance of the top-k documents before passing them to      the LLM.
-Multimodal Integration: Implementing Vision-Language Models (e.g., Llama-3.2 Vision) to allow the assistant to analyze user-uploaded photos of dampness or           structural cracks for better triage.
-Persistence Layer: Connecting the graph to a persistent database (like PostgreSQL) to maintain long-term memory of lead interactions across multiple days or         sessions.
