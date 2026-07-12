# chunk_ids recieved from ChromaDB vector search, 
# graph = persistent graph from memory(loaded with load_graph from graph_store)
# max_nodes = the highest amount of nodes we want to expand to, if the number is >max,
# the nodes after max are ignored.
def expand_chunks(chunk_ids, graph, max_nodes=10):
    expanded = set()
    for cid in chunk_ids:
        expanded.add(cid)
        if cid in graph:
            neighbors = list(graph.neighbors(cid))
            expanded.update(neighbors)

    return list(expanded)[:max_nodes]
