import logging
import chromadb
from pathlib import Path

DEFAULT_COLLECTION_NAME = "documents_collection"
CHROMA_DB_PATH = "./chroma_db"

logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb").setLevel(logging.WARNING)

def create_chroma_client(collection_name=DEFAULT_COLLECTION_NAME):
    try:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name=collection_name)
        return collection
    except Exception as e:
        logging.error(f"Error creating Chroma client: {e}")
        return None

def add_to_chroma(collection, chunks, embeddings, document_metadata=None):
    if not collection or not chunks or embeddings is None:
        logging.error("Missing required parameters for adding to Chroma")
        return False
    ids = [str(i) for i in range(len(chunks))]
    embeddings_list = embeddings.tolist()
    metadatas=[]
    for i in range(len(chunks)):
        metadata={}
        if document_metadata:
            metadata.update(document_metadata)
        metadata["chunk_id"]=i 
        metadatas.append(metadata)   
    try:
        collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
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
        results = collection.query(query_embeddings=query_embedding.tolist(),
                                    n_results=top_k,
                                    include=["documents","metadatas"])
        if not results or "documents" not in results or not results["documents"]:
            logging.error("No results returned from Chroma query")
            return []
        
        contexts=[]
        for i,doc in enumerate(results["documents"][0]):
            metadata=results["metadatas"][0][i]
            doc_name=metadata.get("document_name","Unknown document")
            contexts.append(f"[FROM: {doc_name}]\n{doc}")
        return contexts
    except Exception as e:
        logging.error(f"Error retrieving from Chroma: {e}")
        return []    
    
def empty_collection():
    try:
        collection = create_chroma_client()
        if collection:
            try:
                #deleting the contents only
                all_ids = collection.get(include=[])["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
            except Exception as e:
                try:
                    #deleting the collection itself
                    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                    client.delete_collection(name=DEFAULT_COLLECTION_NAME)
                    client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
                except Exception as nested_e:
                    logging.error(f"Error recreating collection: {nested_e}")
                    return False
            
            logging.info("ChromaDB collection emptied successfully")
            return True
        return False
    except Exception as e:
        logging.error(f"Error emptying collection: {e}")
        return False
