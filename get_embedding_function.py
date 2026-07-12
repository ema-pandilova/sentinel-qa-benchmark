from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return hf_embeddings
