import os
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXTBOOK_DIR = "data/textbooks"
VECTORSTORE_DIR = "data/vectorstore"

# Optional: Set Tesseract path if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

def ocr_pdf(path):
    print(f"--- Starting OCR for {path} ---")
    print("Converting PDF to images (this may take a while)...")
    try:
        images = convert_from_path(path)
    except Exception as e:
        print(f"Error converting PDF to images. Is 'poppler' installed? Error: {e}")
        return []

    docs = []
    total_pages = len(images)
    print(f"Processing {total_pages} pages with OCR...")
    
    for i, image in enumerate(images):
        if i % 5 == 0:
            print(f"OCR Progress: {i}/{total_pages} pages")
        try:
            # Try with Math first (requires 'equ' or specialized training data)
            # Note: Tesseract's 'equ' is often not installed by default or legacy.
            # We will try 'kor+eng' generally, but if user installed 'equ', it helps.
            # However, Tesseract 4/5 doesn't use 'equ' the same way. 
            # Best bet for mixed content is often just 'kor+eng' as standard models handle some symbols.
            # Let's try adding it if available, but safer to default to kor+eng and let the user know.
            
            # Actually, let's keep it simple: 'kor+eng' usually covers basic numbers/symbols.
            # For strict Latex output from Image, specialized models like 'Nougat' are needed.
            # We will assume standard Tesseract usage here.
            text = pytesseract.image_to_string(image, lang='kor+eng') 
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": path, "page": i+1}))
        except Exception as e:
            print(f"OCR Error on page {i+1}: {e}")
            
    return docs

def ingest_textbooks():
    if not os.path.exists(TEXTBOOK_DIR):
        os.makedirs(TEXTBOOK_DIR)
        print(f"Created directory {TEXTBOOK_DIR}. Please add your PDF textbooks there.")
        return

    pdf_files = [f for f in os.listdir(TEXTBOOK_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {TEXTBOOK_DIR}.")
        return

    documents = []
    for pdf_file in pdf_files:
        path = os.path.join(TEXTBOOK_DIR, pdf_file)
        print(f"Loading {path}...")
        
        # 1. Try Standard Loading First
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            
            # Check if meaningful text was extracted
            total_text_len = sum([len(d.page_content.strip()) for d in docs])
            if total_text_len < 100: # Heuristic: if text is too short, assume it's an image
                print(f"Detected image-based PDF (extracted text length: {total_text_len}). Switching to OCR...")
                ocr_docs = ocr_pdf(path)
                if ocr_docs:
                    print(f"OCR Success: Loaded {len(ocr_docs)} pages from {pdf_file}")
                    documents.extend(ocr_docs)
                else:
                    print(f"OCR Failed or no text found for {pdf_file}")
            else:
                print(f"Standard load success: {len(docs)} pages from {pdf_file}")
                documents.extend(docs)
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    if not documents:
        print("No documents loaded.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        print("No text splits created.")
        return

    print(f"Creating vector store from {len(splits)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to {VECTORSTORE_DIR}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_textbooks()
