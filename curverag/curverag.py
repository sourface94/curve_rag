from typing import List, Optional

import numpy as np
from gliner import GLiNER
from sentence_transformers import SentenceTransformer

from curverag.prompts import  PROMPTS
from curverag.atth.train import train
from curverag.graph import create_graph_outlines, create_graph_openai, create_graph_dataset
from curverag.atth.kg_dataset import KGDataset

DEFAULT_ENTITY_TYPES = ["person", "location", "entity", "organisation"]
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

class CurveRAG:

    def __init__(
        self,
        openai_client=None,
        openai_model: str = "gpt-4o-mini",
        llm=None,
        outlines_llm=None,
        entity_types: List[str] = DEFAULT_ENTITY_TYPES,
        gliner_model_name: str = DEFAULT_GLINER_MODEL,
        sentence_transformer_model_name: str=DEFAULT_SENTENCE_TRANSFORMER_MODEL
    ):  
        if openai_client is None and (llm is None or outlines_llm is None):
            raise ValueError("Either an open_ai_client must be provided or both llm and outlines_llm must be provided")
        
        self.using_openai = False
        if openai_client is not None:
            self.openai_client = openai_client
            self.openai_model = openai_model
            self.using_openai = True
        elif llm is not None and outlines_llm is not None:
            self.llm = llm
            self.outlines_llm = outlines_llm
        else:
            raise ValueError("Both llm and outlines_llm must be provdied when using a local LLM")
    
        self.gliner_model = GLiNER.from_pretrained(gliner_model_name)
        self.entity_types = entity_types
        self.graph = None
        self.graph_embedding_model = None
        self.dataset = None
        self.sentence_model = SentenceTransformer(sentence_transformer_model_name)

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
        sentence_transformer_model_name

    ):
        inst = cls(
            llm,
            outlines_llm,
            entity_types,
            gliner_model_name,
            sentence_transformer_model_name
        )
        inst.node_sentence_embeddings = SentenceTransformer(sentence_transformer_model_name).encode([n.name for n in graph.nodes])
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
        if self.using_openai:
            self.graph = create_graph_openai(self.openai_client, docs, model=self.openai_model)
        else:
            self.graph = create_graph(self.outlines_llm, docs)
        print('num nodes', len(self.graph.nodes))
        print('unique node ids', set([n.id for n in self.graph.nodes]))
        print('creating dataset')
        self.dataset = create_graph_dataset(self.graph, dataset_name)

        # TODO: enchance graph with general knowledge that the LLM has - how can we do this?

        # create embeddings
        print('train kg embeddings')
        self.graph_embedding_model = train(self.dataset)
        self.node_sentence_embeddings = self.sentence_model.encode([n.name for n in self.graph.nodes])
        

    def fit(self, dataset: KGDataset):
        """Training of RAGQuery model

        And Mr RAGQuery said: Thou shalt learn the laws of the vocaublary, learn the words and their relation.
        """


        # TODO: enchance graph with general knowledge that the LLM has - how can we do this?

        # create embeddings
        print('train kg embeddings')
        self.graph_embedding_model = train(dataset)        

    def query(self, query: str, additional_entity_types: Optional[List[str]]=None, threshold: float = 0.4, max_tokens: int = 100):

        # get query entites
        if not additional_entity_types:
            additional_entity_types = []
        all_entity_types = self.entity_types + additional_entity_types
        entities = self.gliner_model.predict_entities(query, all_entity_types, threshold=threshold)
        entities = [e['text'] for e in entities]
        print('entities', entities)

        query_embeddings = self.sentence_model.encode(entities + [query])
        similarities = self.sentence_model.similarity(query_embeddings, self.node_sentence_embeddings)
        threshold = 0.5
        similar_indices = [list(np.where(sim_row > threshold)[0]) for sim_row in similarities]
        similar_indices = list(set([i for s in similar_indices for i in s]))
        embedding_idx = [similar_indices]
        print('similar indices', similar_indices)
        nodes_idx_id = {v: k for k, v in self.dataset.nodes_id_idx.items()}
        node_ids = [nodes_idx_id[s] for s in similar_indices]
        print('node_ids', node_ids)
        print('graph nodes retrieved', [n.name for n in self.graph.nodes if n.id in node_ids])
        """
        # get nodes that match entities from graph
        graph_entities = []
        for e in entities:
            for n in self.graph.nodes:
                if n.name.lower() == e.lower():
                    graph_entities.append(n)
        
        # get node embeddings
        embedding_idx = [self.dataset.nodes_id_idx[n.id] for n in graph_entities]
        """
        print('embedding_idx',  embedding_idx, [], len(embedding_idx[0]))
        if len(embedding_idx[0]) == 0:
            print("Found no embedding indx for entities, doing non KGRAG result")
            return "N/A"
        entity_node_embs = self.graph_embedding_model.entity.weight.data[embedding_idx]
        
        #print('entity_node_embs',  entity_node_embs)

        # get all embeddings
        all_node_embs = self.graph_embedding_model.entity.weight.data
        #print('all_node_embs', all_node_embs)

        # traverse graph and get all other related nodes and entities
        similar_node_indexes = self.graph.traverse_hyperbolic_embeddings(entity_node_embs, all_node_embs, threshold=0.6, top_k=5)
        print('similar_node_indexes', similar_node_indexes)
        similar_node_indexes = [n.flatten().tolist() for n in similar_node_indexes]
        similar_node_indexes = [i for s in similar_node_indexes for i in s]
        similar_node_ids = [nodes_idx_id[id] for id in similar_node_indexes]
        print('similar_node_ids', similar_node_ids)
        print('similar_node_ids graph nodes retrieved', [n.name for n in self.graph.nodes if n.id in similar_node_ids])

        node_ids += similar_node_ids 
        node_ids = list(set(node_ids))
                            
        print('node_ids', node_ids)
        query_graph = self.graph.get_subgraph(node_ids)

        # add descriptions of nodes and entities to prompt, along with query 
        prompt_args = {"query": query, "context": str(query_graph)}
        prompt = PROMPTS["generate_response_query_with_references"] # use query and query_graph
        prompt = prompt.format(**prompt_args)
        print(prompt)
        if True:
            return ""
        if self.using_openai:
            result = self.openai_client.responses.create(
                model=self.openai_model,
                input=prompt
            )
            result = result.output_text
        else:
            
            # query llm and return result
            result = self.llm(prompt, max_tokens=max_tokens)
            result = result['choices'][0]['text']

        return result


