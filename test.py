import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from utils_llm import get_llm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VECTORSTORE_DIR = "data/vectorstore"

def test_rag():
    print("--- Testing Textbook RAG ---")
    
    # 1. Initialize Embeddings (Must match ingest.py)
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Load Vector Store
    if not os.path.exists(VECTORSTORE_DIR):
        print(f"Error: {VECTORSTORE_DIR} not found. Please run ingest.py first.")
        return

    try:
        vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    # 3. Interactive Loop
    llm = get_llm()
    
    while True:
        print("\n" + "="*50)
        query = input("질문을 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        # Retrieve
        print("... 교과서 검색 중 ...")
        docs = vectorstore.similarity_search(query, k=5)
        
        context_text = "\n\n".join([f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs])
        
        if not context_text:
            print("교과서에서 관련 내용을 찾을 수 없습니다.")
            continue

        # Generate Answer
        print("... 답변 생성 중 ...")
        system_msg = """당신은 친절한 선생님입니다. 
        학생의 질문에 대해 **반드시 제공된 교과서 내용(Context)**을 바탕으로 답변해주세요.
        교과서 내용을 우선으로 하고, 만약 내용이 부족하다면 그 사실을 언급하고 알고 있는 지식을 덧붙여 설명해주세요.
        답변 끝에는 참고한 교과서 페이지나 출처를 언급해주세요.
        """
        
        user_msg = f"""질문: {query}
        
        [교과서 검색 결과]
        {context_text}
        
        답변:"""
        
        response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
        
        print(f"\n[AI 답변]\n{response.content}")
        print("-" * 20)
        print("[참고한 교과서 내용 일부]")
        for d in docs[:2]:
            print(f"- Page {d.metadata.get('page', '?')}: {d.page_content[:100]}...")

if __name__ == "__main__":
    test_rag()
