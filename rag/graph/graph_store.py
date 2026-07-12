import pickle
import networkx as nx

GRAPH_PATH = "rag/graph/graph.pkl"


def save_graph(graph: nx.Graph, path: str | None = None):
    p = path or GRAPH_PATH
    with open(p, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: str | None = None) -> nx.Graph:
    p = path or GRAPH_PATH
    with open(p, "rb") as f:
        return pickle.load(f)
