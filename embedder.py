import logging
import numpy as np
from fastembed import TextEmbedding
DEFAULT_MODEL_NAME = "BAAI/bge-small-en"

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