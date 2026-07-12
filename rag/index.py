from langchain_chroma import Chroma
from .embeddings import get_embedding_function
from .config import CHROMA_PATH, HNSW_SPACE


def load_db() -> Chroma:
    """
    Load persistent Chroma DB with configured embedding function.
    Make sure you've already populated it with your corpus.
    """
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
        collection_metadata={"hnsw:space": HNSW_SPACE},
    )
