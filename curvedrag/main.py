

def create_graph(text, type='narrative'):

    # chunk text
    text_chunks = text.chunks(chunk_method) # can pass through a chunk method

    graph = Graph()
    for each chunk in text_chunks:
        graph_extraction(graph, chunk)   

def get_response(query: str):

    # embed query
    query = embed_model.encode()

    # classify query type

    # get neighbours
    graph = get_graph()

    # traverse graph
    nodes_and_relationships = graph.traverse()

    result = llm_model.generate(query + nodes_and_relationships)
    return result
