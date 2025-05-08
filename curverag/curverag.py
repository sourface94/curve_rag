from typing import List

from gliner import GLiNER
from sentence_transformers import SentenceTransformer

from curverag.prompts import  PROMPTS
from curverag.atth.train import train
from curverag.graph import create_graph, create_graph_dataset


DEFAULT_ENTITY_TYPES = ["person", "location", "entity", "organisation"]
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"

class CurveRAG:

    def __init__(self, llm, entity_types: List[str] = DEFAULT_ENTITY_TYPES, gliner_model_name: str = DEFAULT_GLINER_MODEL):
        self.llm = llm
        self.gliner_model = GLiNER.from_pretrained(gliner_model_name)
        self.entity_types = entity_types

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
        

    def query(self, query: str, threshold: float = 0.4):

        # classify query type

        # get query entites
        entities = model.predict_entities(query, self.entity_types, threshold=threshold)
        entities = [e['text'] for e in entities]

        # get nodes that match entities from graph
        graph_entities = []
        for e in entities:
            for n in self.graph.nodes:
                if n.name == e:
                    graph_entities.append(n)
        # get node embeddings
        embedding_idx = [self.dataset.nodes_id_idx[n.id] for n in graph_entities]
        entity_node_embs = model.entity.weight.data[embedding_idx]

        # get all embeddings
        all_node_embs = model.entity.weight.data[:10]

        # traverse graph and get all other related nodes and entities
        nodes_and_relationships = self.graph.traverse_hyperbolic_embeddings(entity_node_embs, all_node_embs)

        # add descriptions of nodes and entities to prompt, along with query 
        prompt_args = {"query": query, "context": str(nodes_and_relationships)}
        prompt = PROMPT["generate_response_query_with_references"] # use query and nodes_and_relationships
        prompt = prompt.format(**prompt_args)

        # query llm and return result
        result = self.llm.generate(prompt)

        return result



