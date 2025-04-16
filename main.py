import os
import logging
from pathlib import Path
import numpy as np
import PyPDF2
import chromadb
from fastembed import TextEmbedding
import google.generativeai as genai
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb").setLevel(logging.WARNING)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
else:
    genai.configure(api_key=API_KEY)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_COLLECTION_NAME = "documents_collection"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en"
CHROMA_DB_PATH = "./chroma_db"
UPLOADS_DIR = "uploads"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except FileNotFoundError:
        logging.error(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None
    
def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    if not text:
        logging.warning("No text provided for chunking")
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size 
        chunks.append(text[start:end])
        start = end - chunk_overlap
    logging.info(f"Text split into {len(chunks)} chunks")
    return chunks

def create_embeddings_fastembed(chunks, model_name=DEFAULT_MODEL_NAME):
    if not chunks:
        logging.warning("No chunks provided for embedding creation")
        return None
    try:
        model = TextEmbedding(model_name=model_name)
        embeddings = list(model.embed(chunks))
        return np.array(embeddings)
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return None
    
def generate_answer(query, context):
    if not API_KEY:
        return "ERROR: API key for Gemini is not configured."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Based on the following context, please answer the question accurately. 
                    If the answer cannot be found in the context, politely indicate this.
                    Context: {context}
                    Question: {query}
                    Answer:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return f"Sorry, couldn't generate an answer. Error: {str(e)}"
    
def create_chroma_client(collection_name=DEFAULT_COLLECTION_NAME):
    try:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name=collection_name)
        return collection
    except Exception as e:
        logging.error(f"Error creating Chroma client: {e}")
        return None

def add_to_chroma(collection, chunks, embeddings):
    if not collection or not chunks or embeddings is None:
        logging.error("Missing required parameters for adding to Chroma")
        return False
    ids = [str(i) for i in range(len(chunks))]
    embeddings_list = embeddings.tolist()
    try:
        collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            ids=ids
        )
        logging.info(f"Added {len(chunks)} chunks to ChromaDB")
        return True
    except Exception as e:
        logging.error(f"Error adding to Chroma: {e}")
        return False

def retrieve_from_chroma(collection, query_embedding, top_k=3):
    if collection is None or query_embedding is None:
        logging.error("Collection or query embedding is None")
        return []
    try:
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logging.error(f"Error retrieving from Chroma: {e}")
        return []

def ensure_uploads_directory():
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR

def delete_uploaded_files():
    try:
        uploads_dir = ensure_uploads_directory()
        deleted_count = 0
        for file in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1
        logging.info(f"Deleted {deleted_count} files from uploads directory")
        return True
    except Exception as e:
        logging.error(f"Error deleting uploaded files: {e}")
        return False

def empty_collection():
    try:
        collection = create_chroma_client()
        if collection:
            collection.delete()
            logging.info("ChromaDB collection emptied successfully")
            return True
        return False
    except Exception as e:
        logging.error(f"Error emptying collection: {e}")
        return False

###################################################### for module specific testing below v ###########################
def initialize_rag_chroma_pipeline(pdf_path):
    logging.info("Extracting text from uploaded PDF...")
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        return False, "Failed to load PDF."
    logging.info("Splitting text into chunks...")
    chunks = chunk_text(text)
    if not chunks:
        return False, "Failed to create chunks."
    logging.info("Creating vector embeddings for the chunks...")
    embeddings = create_embeddings_fastembed(chunks)
    if embeddings is None:
        return False, "Failed to create embeddings."
    logging.info("Creating a ChromaDB client...")
    chroma_collection = create_chroma_client()
    if chroma_collection is None:
        return False, "Failed to create ChromaDB collection."
    logging.info("Storing the embeddings into database...")
    if not add_to_chroma(chroma_collection, chunks, embeddings):
        return False, "Failed to add data to ChromaDB."
    return True, "RAG pipeline initialized successfully."
    
def rag_pipeline_chroma(query):
    logging.info("Creating embeddings for user query...")
    try:
        query_embedding = list(TextEmbedding().embed([query]))[0]
    except Exception as e:
        logging.error(f"Error creating query embedding: {e}")
        return "Failed to process your query. Please try again."
    logging.info("Checking database for relevant info...")
    collection = create_chroma_client()
    if collection is None:
        return "Database connection error. Please try again later."
    relevant_chunks = retrieve_from_chroma(collection, np.array(query_embedding), top_k=3)
    if not relevant_chunks:
        return "I couldn't find relevant information in the document to answer your question."
    context = "\n".join(relevant_chunks)
    logging.info("LLM is generating response...")
    answer = generate_answer(query, context)
    return answer

if __name__ == "__main__":
    sample_pdf_path = input("")
    
    if os.path.exists(sample_pdf_path):
        success, message = initialize_rag_chroma_pipeline(sample_pdf_path)
        print(message)
        
        if success:
            print("\nRAG pipeline initialized successfully.")
            print("You can ask questions about the document. Type 'exit' to quit.\n")
            
            while True:
                query = input("Ask a question ('exit' to quit): ")
                if query.lower() == 'exit':
                    break
                answer = rag_pipeline_chroma(query)
                print("\nAnswer:", answer, "\n")

        print("Cleaning up resources...")
        empty_collection()
    else:
        print(f"File not found: {sample_pdf_path}")