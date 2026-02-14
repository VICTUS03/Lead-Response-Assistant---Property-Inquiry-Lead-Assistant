# Lead-Response-Assistant---Property-Inquiry-Lead-Assistant

Run the App:streamlit run main.py

# 🏠 Property Lead Response Assistant

An intelligent, stateful AI workflow designed to convert raw customer inquiries into professional, grounded, and safe property diagnostic responses. This system moves beyond simple prompting by using **LangGraph** for logic control and **FAISS** for Retrieval-Augmented Generation (RAG).

---

## 🚀 Key Features
* **Intelligent Intent Analysis:** Automatically categorizes inquiries (e.g., Dampness, Leaks) and assesses urgency.
* **Grounded Truth (RAG):** Uses a FAISS vector store to ensure advice follows official safety protocols.
* **Anti-Hallucination Guardrails:** Explicitly engineered to avoid making false promises or giving unauthorized cost estimates.
* **Real-time Process Monitoring:** A built-in Streamlit dashboard that visualizes the AI's internal "thinking" process.

---

## 🛠️ Technical Stack
* **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) (Stateful Directed Acyclic Graph)
* **LLM:** [Groq](https://groq.com/) (Llama-3-70b) for high-speed, high-reasoning inference.
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Local index)
* **Embeddings:** `all-MiniLM-L6-v2` via HuggingFace (Local & Lightweight)
* **Interface:** Streamlit

---

## 🧠 How the System Works
The assistant follows a structured **Stateful Workflow**:

1.  **Analyze Intent Node:** Identifies the core issue and extracts keywords (e.g., "Bedroom," "Heavy rain").
2.  **Retrieve Knowledge Node:** Queries the FAISS index to find relevant safety guidelines and clarifying questions.
3.  **Draft Response Node:** Combines the user query with retrieved "Grounded Truth" to generate a professional, empathetic reply.



---

## ✅ Accuracy & Reliability
To ensure the output is "Client-Ready," the system implements:
* **Temperature=0:** Ensures deterministic, professional responses without "creative" hallucinations.
* **Negative Prompting:** Strict constraints prevent the AI from quoting prices or guaranteeing fixes.
* **Source Grounding:** The AI is instructed to prioritize the safety context retrieved from the vector store over its general training data.

---

## ⚠️ Known Limitations
* **Text-Only Context:** The current system lacks computer vision and cannot analyze photos of damage.
* **Static Index:** The knowledge base is currently loaded from a local file rather than a live-syncing database.
* **Semantic Collisions:** Extremely vague queries may occasionally pull broadly related instead of highly specific protocols.

---

## 🔮 Future Improvements
* **Cross-Encoder Reranking:** Adding a secondary reranking step after FAISS to maximize the semantic relevance of retrieved safety protocols.
* **Multimodal Analysis:** Integrating Vision-Language Models (e.g., Llama-3.2 Vision) to allow the AI to "see" and triage dampness patterns.
* **Postgres Persistence:** Implementing long-term memory to maintain lead context across multiple days or sessions.
* **Dynamic Knowledge Pipeline:** Connecting the FAISS store to a live CRM or technical documentation API.

---

