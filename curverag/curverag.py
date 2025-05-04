from typing import List

from sentence_transformers import SentenceTransformer

from curverag.prompts import  PROMPTS
from curverag.atth.train import train
from curverag.graph import create_graph, create_graph_dataset


class CurveRAG:

    def __init__(self, llm):
        self.llm = llm

    def fit(self, docs: List[str], dataset_name: str):
        """Training of RAGQuery model

        And Mr RAGQuery said: Thou shalt learn the laws of the vocaublary, learn the words and their relation.
        """

        # create graph
        print('creating graph')
        self.graph = create_graph(self.llm, docs)
        print('creating dataset')
        self.dataset = create_graph_dataset(self.graph, dataset_name)

        # create embeddings
        print('dataset type', type(self.dataset))
        print('train kg embeddings')
        self.model = train(self.dataset)
        

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



