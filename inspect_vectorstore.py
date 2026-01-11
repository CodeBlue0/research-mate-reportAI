from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTORSTORE_DIR = "data/vectorstore"

def inspect_vectorstore():
    print(f"--- Inspecting Vector Store in {VECTORSTORE_DIR} ---")
    
    if not os.path.exists(VECTORSTORE_DIR):
        print("Vector store not found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # FAISS doesn't allow easy iteration over all docs, but we can access the docstore
    docstore = vectorstore.docstore._dict
    
    print(f"Total chunks stored: {len(docstore)}")
    print("\n--- Sample Chunks (First 5) ---\n")
    
    for i, (doc_id, doc) in enumerate(docstore.items()):
        if i >= 5: break
        print(f"[Chunk {i+1}] Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
        print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
        print("-" * 50)

if __name__ == "__main__":
    inspect_vectorstore()
