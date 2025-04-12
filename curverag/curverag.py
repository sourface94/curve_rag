from sentence_transformers import SentenceTransformer

from curverag.graph import Graph
from curverag.prompts import PROMPT


class GraphRAG:

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
        prompt = PROMPT["generate_response_query_with_references"] # use query and nodes_and_relationships
        prompt_args = {"query": query, "context": str(nodes_and_relationships)}
        prompt = prompt.format(**prompt_args)
        result = self.llm.generate(prompt)
        return result
