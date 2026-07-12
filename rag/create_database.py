import argparse
import os
import shutil

from embeddings import get_embedding_function
from config import CHROMA_PATH
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "backend", "merged_pdf", "batch6")


def main():
    clear_database()
    generate_data_store()


def clear_database():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)


def generate_data_store():
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)


def load_documents():
    # Load merged_pdf from the specified directory DATA_PATH
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents


def split_documents(documents: list[Document]):
    # Split the pdfs into smaller chunks for better retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["chunk_id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")



# This will create IDs like "merged_pdf/merged_pdfs.pdf:6:2"
# Page Source : Page Number : Chunk:Index
def calculate_chunk_ids(chunks):
    """
    Assign stable, semantic IDs to each chunk.

    chunk_id: source:page:chunk_index
    doc_id:   source document path
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")

        if source is None or page is None:
            raise ValueError("Missing source or page in chunk metadata")

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["doc_id"] = source

    return chunks


if __name__ == "__main__":
    main()
