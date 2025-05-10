from typing import List

from gliner import GLiNER
from sentence_transformers import SentenceTransformer

from curverag.prompts import  PROMPTS
from curverag.atth.train import train
from curverag.graph import create_graph, create_graph_dataset


DEFAULT_ENTITY_TYPES = ["person", "location", "entity", "organisation"]
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"

class CurveRAG:

    def __init__(self, llm, outlines_llm, entity_types: List[str] = DEFAULT_ENTITY_TYPES, gliner_model_name: str = DEFAULT_GLINER_MODEL):
        self.llm = llm
        self.outlines_llm = outlines_llm
        self.gliner_model = GLiNER.from_pretrained(gliner_model_name)
        self.entity_types = entity_types
        self.graph = None
        self.graph_embedding_model = None
        self.dataset = None

    @classmethod
    def load_class(
        cls,
        llm,
        outlines_llm,
        entity_types: List[str],
        gliner_model_name: str,
        graph,
        graph_embedding_model,
        dataset,

    ):
        inst = cls(
            llm,
            outlines_llm,
            entity_types,
            gliner_model_name
        )
        inst.graph = graph
        inst.graph_embedding_model = graph_embedding_model
        inst.dataset = dataset
        return inst

        

    def fit(self, docs: List[str], dataset_name: str):
        """Training of RAGQuery model

        And Mr RAGQuery said: Thou shalt learn the laws of the vocaublary, learn the words and their relation.
        """

        # create graph
        print('creating graph')
        self.graph = create_graph(self.llm, docs)
        print('creating dataset')
        self.dataset = create_graph_dataset(self.graph, dataset_name)

        # enchance graph with general knowledge that the LLM has - how can we do this?

        # create embeddings
        print('dataset type', type(self.dataset))
        print('train kg embeddings')
        self.graph_embedding_model = train(self.dataset)
        

    def query(self, query: str, threshold: float = 0.4, max_tokens: int = 100):

        # get query entites
        entities = self.gliner_model.predict_entities(query, self.entity_types, threshold=threshold)
        entities = [e['text'] for e in entities]
        print('entities', entities)
        #entities = ['Patient', 'Diabetes Mellitus', 'Metformin', 'Type 2 Diabetes', 'Daily']

        # get nodes that match entities from graph
        graph_entities = []
        for e in entities:
            for n in self.graph.nodes:
                if n.name.lower() == e.lower():
                    graph_entities.append(n)
        
        # get node embeddings
        embedding_idx = [self.dataset.nodes_id_idx[n.id] for n in graph_entities]
        entity_node_embs = self.graph_embedding_model.entity.weight.data[embedding_idx]

        # get all embeddings
        all_node_embs = self.graph_embedding_model.entity.weight.data[:10]

        # traverse graph and get all other related nodes and entities
        node_ids = self.graph.traverse_hyperbolic_embeddings(entity_node_embs, all_node_embs)
        print('node_ids', node_ids)
        node_ids = [n.flatten().tolist() for n in node_ids]
        node_ids = [i for s in node_ids for i in s]

        node_ids
        print('node_ids', node_ids)
        query_graph = self.graph.get_subgraph(list(set(node_ids+[n.id for n in graph_entities])))

        # add descriptions of nodes and entities to prompt, along with query 
        prompt_args = {"query": query, "context": str(query_graph)}
        prompt = PROMPTS["generate_response_query_with_references"] # use query and query_graph
        prompt = prompt.format(**prompt_args)
        print(prompt)
        # query llm and return result
        result = self.llm(prompt, max_tokens=max_tokens)

        return result['choices'][0]['text']



