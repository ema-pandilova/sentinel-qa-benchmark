
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)
