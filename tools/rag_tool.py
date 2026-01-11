from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTORSTORE_DIR = "data/vectorstore"

def query_textbook(query: str) -> str:
    """Useful for finding information in the provided textbooks."""
    if not os.path.exists(VECTORSTORE_DIR):
        return "No textbook data found. Please run ingest.py first."
    
    # embeddings = OpenAIEmbeddings() # Deprecated for DeepSeek usage
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])
