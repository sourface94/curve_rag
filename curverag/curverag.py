from sentence_transformers import SentenceTransformer

from curverag.graph import Graph
from curverag.prompts import PROMPT
from curverag.atth.train


class RAGQuery:

    def __init__(self, ll, dataset_name: str, embedding_model: str):
        self.embed_model = SentenceTransformer(embedding_model)
        self.llm = ''
        self.dataset_name = dataset_name

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

    def fit(dataset: List[str]):
        """Training of RAGQuery model

        And Mr RAGQuery said: Thou shalt learn the laws of the vocaublary, learn the words and their relation.
        """

        # create graph
        graph = create_graph(self.llm, dataset)
        dataset = create_graph_dataset(graph, )

        # create embeddings
        train.train()



