from sentence_transformers import SentenceTransformer

from curverag.graph import Graph


class CurveRAG:

    def __init__(self, graph: Graph, embedding_model: str):
        self.embed_model = SentenceTransformer(embedding_model)
        self.graph = graph
        self.llm = ''

    def query(self, query: str):

        # embed query
        query = self.embed_model.encode()

        # classify query type

        # traverse graph
        nodes_and_relationships = self.graph.traverse(query)
        prompt = '' # use query and nodes_and_relationships
        result = self.llm.generate(prompt)
        return result