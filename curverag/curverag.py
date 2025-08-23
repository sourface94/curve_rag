from typing import List, Optional

import numpy as np
import spacy
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
        sentence_transformer_model_name: str=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        spacy_model: str="en_core_web_lg"
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
        self.spacy_nlp = spacy.load(spacy_model)

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
    
    def generate_node_embeddings(self):
        print('graph embeddings done')
        self.node_sentence_embeddings = self.sentence_model.encode([n.name for n in self.graph.nodes])
        self.edge_sentence_embeddings = self.sentence_model.encode([self.get_edge_description(self.graph, e) for e in self.graph.edges])

    def fit(self, docs: List[str], dataset_name: str):
        """Training of RAGQuery model

        And Mr RAGQuery said: Thou shalt learn the laws of the vocaublary, learn the words and their relation.
        """

        # create graph
        print('creating graph')
        if self.using_openai:
            self.graph = create_graph_openai(self.openai_client, docs, model=self.openai_model)
        else:
            self.graph = create_graph_outlines(self.outlines_llm, docs)
        print('num nodes', len(self.graph.nodes))
        print('unique node ids', set([n.id for n in self.graph.nodes]))
        print('creating dataset')
        self.dataset = create_graph_dataset(self.graph, dataset_name)

        # TODO: enchance graph with general knowledge that the LLM has - how can we do this?

        # create embeddings
        print('train kg embeddings')
        self.graph_embedding_model = train(self.dataset)
        print('get node sentence embeddings')
        self.node_sentence_embeddings = self.sentence_model.encode([n.name for n in self.graph.nodes])
        print('get edge sentence embeddings')
        self.edge_sentence_embeddings = self.sentence_model.encode([self.get_edge_description(self.graph, e) for e in self.graph.edges])

    def fit_(self, dataset: KGDataset):
        """Training of RAGQuery model
        """
        # TODO: enchance graph with general knowledge that the LLM has - how can we do this?

        # create embeddings
        print('train kg embeddings')
        self.graph_embedding_model = train(dataset) 
        print(type(self.graph_embedding_model))    
        self.node_sentence_embeddings = self.sentence_model.encode([n.name for n in self.graph.nodes])
        self.edge_sentence_embeddings = self.sentence_model.encode([self.get_edge_description(self.graph, e) for e in self.graph.edges])   

    def get_query_entities(self, query, threshold, additional_entity_types):
        # get gliner entites
        if not additional_entity_types:
            additional_entity_types = []
        all_entity_types = self.entity_types + additional_entity_types
        entities = self.gliner_model.predict_entities(query, all_entity_types, threshold=threshold)
        entities = [e['text'] for e in entities]
        print('entities', entities)

        query_sp = self.spacy_nlp(query)
        pos_tags = []
        for token in query_sp:
            print(token.text, token.pos_, token.tag_)
            if token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                pos_tags.append(token.text)
        print('pos_tags', pos_tags)
        entities += pos_tags
        return entities
    
    def get_edge_description(self, graph, edge):
        return graph.get_matching_node_by_id(edge.source).name + " has a relationship with " + graph.get_matching_node_by_id(edge.target).name + " called " + edge.name + " and desribed as: " + edge.description 

    def get_edge_types(self, entities):
        print('entites for edges', entities)
        entities = self.sentence_model.encode(entities)
        edges = self.sentence_model.encode([e.name for e in self.graph.edges])
        similarities = self.sentence_model.similarity(edges, entities)
        #print('similarities', similarities)
        threshold = 0.5
        similar_indices = [list(np.where(sim_row > threshold)[0]) for sim_row in similarities]
        similar_indices = list(set([i for s in similar_indices for i in s]))
        similar_edges = [e.name for i, e in enumerate(self.graph.edges) if i in similar_indices]
        print('similar_edges', similar_edges)
        return similar_edges
    
    def query(self,
        query: str,
        additional_entity_types: Optional[List[str]]=None,
        threshold: float = 0.4,
        edge_threshold = 0.5,
        max_tokens: int = 100,
        traversal='hyperbolic',
        top_k: int = 10
    ):

        entities = self.get_query_entities(query, threshold, additional_entity_types)

        # get all query nodes using sentence transformer
        query_embeddings = self.sentence_model.encode(entities + [query])
        similarities = self.sentence_model.similarity(query_embeddings, self.node_sentence_embeddings)
        similar_indices = [list(np.where(sim_row > threshold)[0]) for sim_row in similarities]
        similar_indices = list(set([i for s in similar_indices for i in s]))
        
        # map node ids to those in the KnowldgeGraph (self.graph)
        nodes_idx_id = {v: k for k, v in self.dataset.nodes_id_idx.items()}
        node_ids = [nodes_idx_id[s] for s in similar_indices]
        print('graph nodes retrieved', [n.name for n in self.graph.nodes if n.id in node_ids])

        # get addditonal query nodes from edges using sentence transformer 
        similarities = self.sentence_model.similarity(query_embeddings, self.edge_sentence_embeddings)
        print('similarities', similarities)
        edge_similar_indices = [list(np.where(sim_row > edge_threshold)[0]) for sim_row in similarities]
        print('edge_similar_indices', edge_similar_indices)
        edge_similar_indices = list(set([i for s in edge_similar_indices for i in s]))
        print('edge_similar_indices', edge_similar_indices)
        for i in edge_similar_indices: # get edge node ids and and idx
            edge = self.graph.edges[i]
            print('edge to add', edge)
            node_ids.append(edge.source)
            similar_indices.append(self.dataset.nodes_id_idx[edge.source])

            node_ids.append(edge.target)
            similar_indices.append(self.dataset.nodes_id_idx[edge.target])

        similar_indices = list(set(similar_indices))
        node_ids = list(set(node_ids))
        if len(similar_indices) == 0:
            print("Found no embedding indx for entities, doing non KG-RAG result")
            return "N/A"

        # get related nodes by trraversing through graph
        if traversal == 'hyperbolic':
            embedding_idx = [similar_indices]
            entity_node_embs = self.graph_embedding_model.entity.weight.data[embedding_idx]
            # get all embeddings
            all_node_embs = self.graph_embedding_model.entity.weight.data
            # print('all_node_embs', all_node_embs)
            # traverse graph and get all other related nodes and entities
            similar_node_indexes = self.graph.traverse_hyperbolic_embeddings(entity_node_embs, all_node_embs, threshold=0.6, top_k=top_k)
            similar_node_indexes = [int(i) for s in similar_node_indexes for i in s]
            similar_node_indexes = list(set(similar_node_indexes))
            similar_node_ids = [nodes_idx_id[id] for id in similar_node_indexes]
        elif traversal == 'pp':
            edge_types = self.get_edge_types(entities)
            similar_node_ids = self.graph.traverse_personalised_pagerank(node_ids, top_k=top_k, edge_types=edge_types)
        print('similar_node_ids', similar_node_ids)
        print('similar_node_ids graph nodes retrieved', [n.name for n in self.graph.nodes if n.id in similar_node_ids])
          
        # add nodes retrieved from grpah rtarversal to all nodes that will be added to prompy
        node_ids += similar_node_ids 
        node_ids = list(set(node_ids))

        # create a subgraph including all nodes we want to add to prompt                            
        query_graph = self.graph.get_subgraph(node_ids)

        # add descriptions of nodes and entities to prompt, along with query 
        prompt_args = {"query": query, "context": str(query_graph)}
        prompt = PROMPTS["generate_response_query_with_references"] # use query and query_graph
        prompt = prompt.format(**prompt_args)
        print(prompt)

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
