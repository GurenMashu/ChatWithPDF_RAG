import streamlit as st
import os
from datetime import datetime
import tempfile
from pathlib import Path
import numpy as np
import logging
from fastembed import TextEmbedding

from main import (
    extract_text_from_document,
    chunk_text,
    create_embeddings_fastembed,
    create_chroma_client,
    add_to_chroma,
    retrieve_from_chroma,
    generate_answer,
    empty_collection,
    ensure_uploads_directory
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb").setLevel(logging.WARNING)

def main():
    st.set_page_config(page_title="Document Chat Assistant", page_icon="📚", layout="wide")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "documents" not in st.session_state:
        st.session_state.documents=[]    
    
    with st.sidebar:
        st.title("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv"],
                                            help="Supported formats: PDF, DOCX, TXT, CSV")
        
        if uploaded_file is not None:
            # Creating a temporary file
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    success = process_document(path)
                    if success:
                        st.session_state.document_processed = True
                        st.session_state.documents.append(uploaded_file.name)
                        st.success(f"Document '{uploaded_file.name}' processed successfully!")
                    else:
                        st.error("Failed to process document. Please try again.")
        if st.session_state.documents:
            st.subheader("Processed Documents")
            for doc in st.session_state.documents:
                st.write(f"-{doc}")

        st.divider()  

        st.subheader("Database Management")
        
        if st.button("Clear Document Data", type="secondary", use_container_width=True):
            if empty_collection():
                st.session_state.document_processed = False
                st.session_state.messages = []
                st.session_state.documents = []
                st.success("Document data cleared successfully!")
            else:
                st.error("Failed to clear document data.")               
        
        st.divider()
        st.markdown("### How to use")
        st.markdown("""
        1. Upload a document
        2. Click [Process Document]
        3. Ask questions about the contents in the chat!
        """)
    
    st.title("📚 Chat with Documents")
    
    if not st.session_state.document_processed:
        st.info("Please upload a document using the sidebar to start chatting.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = get_answer(prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
################################################## END OF UI STUFFS ^ ###################################################

def process_document(pdf_path):
    """Process the uploaded document with visual feedback"""
    with st.status("Processing document...", expanded=True) as status:
        st.write("Extracting text from uploaded document...")
        text = extract_text_from_document(pdf_path)
        if text is None:
            status.update(label="Failed to load document", state="error")
            return False
        
        doc_name=os.path.basename(pdf_path)
        doc_id=f"doc_{hash(doc_name)}_{int(os.path.getmtime(pdf_path))}"
        document_metadata={
            "document_id":doc_id,
            "document_name":doc_name,
            "source_path":pdf_path,
            "created_at":str(os.path.getctime(pdf_path)),
            "modified_at":str(os.path.getmtime(pdf_path))
        }

        st.write("Splitting text into chunks...")
        chunks = chunk_text(text)
        if not chunks:
            status.update(label="Failed to create chunks", state="error")
            return False
        
        st.write("Creating vector embeddings for the chunks...")
        embeddings = create_embeddings_fastembed(chunks)
        if embeddings is None:
            status.update(label="Failed to create embeddings", state="error")
            return False
        
        st.write("Creating a ChromaDB client...")
        chroma_collection = create_chroma_client()
        if chroma_collection is None:
            status.update(label="Failed to create ChromaDB collection", state="error")
            return False
        
        st.write("Storing the embeddings into database...")
        if not add_to_chroma(chroma_collection, chunks, embeddings, document_metadata):
            status.update(label="Failed to add data to ChromaDB", state="error")
            return False
            
        status.update(label="Document processed successfully!", state="complete")
        return True

def get_answer(query):
    """Get answer for user query with visual feedback"""
    with st.status("Processing your question...", expanded=True) as status:
        try:
            st.write("Creating embeddings for user query...")
            query_embedding = list(TextEmbedding().embed([query]))[0]
            
            st.write("Checking database for relevant info...")
            # fresh connection 
            collection = create_chroma_client()
            if collection is None:
                status.update(label="Failed to connect to database", state="error")
                return "Failed to connect to the document database."
            
            relevant_chunks = retrieve_from_chroma(collection, np.array(query_embedding), top_k=3)
            if not relevant_chunks:
                status.update(label="No relevant information found", state="complete")
                return "No relevant information found in the document to answer your question."
            
            context = "\n".join(relevant_chunks)
            st.write("Generating response...")
            answer = generate_answer(query, context)
            status.update(label="Response ready!", state="complete")
            return answer
            
        except Exception as e:
            logging.error(f"Error in get_answer: {e}")
            status.update(label="Error processing question", state="error")
            return f"Sorry, an error occurred while processing your question: {str(e)}"

if __name__ == "__main__":
    ensure_uploads_directory()
    main()