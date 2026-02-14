import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document   

# This ensures the AI follows the objective: Acknowledge, Ask, Safe Steps, No Promises, basically it define the rules or requirements the llm should consider while generating an resopnse.
KNOWLEDGE_DATA = [
    {"content": "When a customer reports dampness or leaks, always ask about the age of the property and if it happens specifically during rain.", "metadata": {"category": "clarifying_questions"}},
    {"content": "Safe next steps for dampness: Suggest checking gutters for blockages and moving furniture away from the affected wall to allow airflow.", "metadata": {"category": "advice"}},
    {"content": "Never provide a cost estimate or a guaranteed fix over chat. Always state that an on-site professional inspection is required for a definitive solution.", "metadata": {"category": "safety"}},
    {"content": "Identify if the issue is urgent (e.g., water near electrical sockets) and advise the customer to turn off power if safety is at risk.", "metadata": {"category": "emergency"}},
    {"content": "Standard procedure: Acknowledge the stress of the situation, ask for clear photos of the issue, and explain that our team needs to see the damp pattern.", "metadata": {"category": "workflow"}},
    {
    "content": "Standard procedure: If the user query is unrelated to building health, dampness, leaks, or structural issues, provide the standard refusal: 'Please ask property related questions.'",
    "metadata": {"category": "out_of_scope_rule"}}
]

class VectorStoreManager:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def create_and_save_index(self):
        """Converts raw knowledge into a searchable FAISS index and stores in it."""
        docs = [Document(page_content=item["content"], metadata=item["metadata"]) for item in KNOWLEDGE_DATA]
        
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.index_path)
        print(f"Vector store saved to {self.index_path}")

    def load_index(self):
        """Loads the index from disk."""
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.create_and_save_index()

    def search(self, query, k=3):
        """Retrieves the top 3 most relevant safety/procedure documents."""
        if not self.vector_store:
            self.load_index()
        return self.vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Test run
    manager = VectorStoreManager()
    manager.create_and_save_index()
    results = manager.search("I have damp patches in my room")
    for res in results:
        print(f"\n[Retrieved]: {res.page_content}")