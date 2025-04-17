import streamlit as st
import os
from datetime import datetime
import tempfile
import base64
import numpy as np
import logging
from fastembed import TextEmbedding

from main import (
    extract_text_from_document,
    chunk_text,
    create_embeddings_fastembed,
    generate_answer,
    ensure_uploads_directory
)
from infodownload import create_download_link
from DBfuncs import create_chroma_client, add_to_chroma, retrieve_from_chroma, empty_collection

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
    if "selected_responses" not in st.session_state:
        st.session_state.selected_responses=[]        
    
    with st.sidebar:
        st.title("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv"],
                                            help="Supported formats: PDF, DOCX, TXT, CSV")
        
        if uploaded_file is not None:
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

        if st.session_state.selected_responses:
            st.subheader("Save selected Responses")
            if st.button("Download selected responses", use_container_width=True):
                download_link=create_download_link(st.session_state.selected_responses)
                st.markdown(download_link, unsafe_allow_html=True)

            if st.button("Clear selected responses", type="secondary", use_container_width=True):
                st.session_state.selected_responses=[]
                st.success("Selected responses cleared.")     

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
    
    st.title("✨ Chat with Documents")
    
    if not st.session_state.document_processed:
        st.info("Please upload a document using the sidebar to start chatting.")
    else:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Add checkbox next to assistant messages
                if message["role"] == "assistant":
                    # Create a unique key for each checkbox
                    checkbox_key = f"select_response_{i}"
                    
                    # Check if this response is already in selected_responses
                    response_details = {
                        "question": st.session_state.messages[i-1]["content"] if i > 0 else "",
                        "answer": message["content"],
                        "timestamp": message.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    
                    # Check if this response is in selected_responses list
                    is_selected = any(
                        r["answer"] == response_details["answer"] and 
                        r["question"] == response_details["question"]
                        for r in st.session_state.selected_responses
                    )
                    
                    # Display checkbox with the current state
                    if st.checkbox("Save this response", key=checkbox_key, value=is_selected):
                        # Add to selected responses if not already there
                        if not is_selected:
                            st.session_state.selected_responses.append(response_details)
                    else:
                        # Remove from selected responses if it was there
                        if is_selected:
                            st.session_state.selected_responses = [
                                r for r in st.session_state.selected_responses 
                                if not (r["answer"] == response_details["answer"] and 
                                       r["question"] == response_details["question"])
                            ]
        
        if prompt := st.chat_input("Ask a question about your document"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = get_answer(prompt)
                st.markdown(response)

                new_response_key=f"select_response_{len(st.session_state.messages)}"
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if st.checkbox("Save this response", key=new_response_key):
                    response_details-{
                        "question":prompt,
                        "answer":response,
                        "timestamp":timestamp
                    }
                    st.session_state.selected_responses.append(response_details)
            st.session_state.messages.append({
                "role":"assistant",
                "content":response,
                "timestamp":timestamp
            })        
################################################## END OF UI STUFFS ^ ###################################################

def process_document(pdf_path):
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
            return f"Sorry, an error processing your question: {str(e)}"

if __name__ == "__main__":
    ensure_uploads_directory()
    main()