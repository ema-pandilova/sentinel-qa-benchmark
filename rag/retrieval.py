from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document

from .index import load_db
from .config import DEFAULT_K
# added for needs of graph
def fetch_docs_by_ids(ids: list[str]):
    """
    Retrieve Document objects from Chroma by their chunk IDs.
    """
    if not ids:
        return []
    db = load_db()

    results = db.get(ids=ids)  # returns dict with 'documents' and 'metadatas'
    
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    return documents

def retrieve_context(
    query_text: str,
    k: int = DEFAULT_K,
    return_scores: bool = False,
) -> Optional[str | List[Tuple[Document, float]]]:
    """
    Retrieve top-k documents (with scores) or only the joined text context.
    """
    db = load_db()
    results: List[Tuple[Document, float]] = db.similarity_search_with_score(
        query_text, k=k
    )

    if return_scores:
        return results

    if not results:
        return None

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    return context_text


def docs_to_metadata(
    docs_with_scores: List[Tuple[Document, float]]
) -> List[Dict[str, Any]]:
    """
    Convert retrieved docs to a simple metadata list for logging and metrics.
    Expect metadata to contain at least: doc_id, chunk_id, source (you control this when indexing).
    """
    meta_list: List[Dict[str, Any]] = []
    for doc, score in docs_with_scores:
        # Chroma stores chunk_id in "id" field and doc_id in "source" field
        meta_list.append(
            {
                "doc_id": doc.metadata.get("doc_id") or doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id") or doc.metadata.get("id"),
                "source": doc.metadata.get("source"),
                "score": score,
            }
        )
    return meta_list
