import argparse
from collections import defaultdict

import networkx as nx
import numpy as np
from langchain_chroma import Chroma
from sklearn.metrics.pairwise import cosine_similarity

from rag.config import CHROMA_PATH
from rag.embeddings import get_embedding_function
from rag.graph.graph_store import save_graph

DEFAULT_SIMILARITY_THRESHOLD = 0.8


def build_graph(
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    graph_path: str | None = None,
):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    data = db.get(include=["metadatas", "embeddings"])
    metadatas = data["metadatas"]
    ids = data["ids"]

    G = nx.Graph()

    for chunk_id in ids:
        G.add_node(chunk_id)

    # Same-page edges
    pages = defaultdict(list)
    for cid, meta in zip(ids, metadatas):
        key = (meta.get("source"), meta.get("page"))
        pages[key].append(cid)

    for chunks in pages.values():
        for i in range(len(chunks) - 1):
            G.add_edge(chunks[i], chunks[i + 1], type="same_page")

    # Similarity edges
    vectors = np.array(data["embeddings"])
    sim_matrix = cosine_similarity(vectors)

    for i, cid1 in enumerate(ids):
        for j, cid2 in enumerate(ids):
            if i >= j:
                continue
            sim = sim_matrix[i, j]
            if sim >= similarity_threshold:
                G.add_edge(cid1, cid2, type="similarity", weight=sim)

    save_graph(G, path=graph_path)
    print(
        f"Graph built: threshold={similarity_threshold}, "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    ap.add_argument("--output", type=str, default=None,
                    help="Output path (default: rag/graph/graph_t{threshold}.pkl)")
    args = ap.parse_args()
    path = args.output or f"rag/graph/graph_t{args.threshold}.pkl"
    build_graph(similarity_threshold=args.threshold, graph_path=path)
