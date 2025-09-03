import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import spacy
from gliner import GLiNER
from sentence_transformers import SentenceTransformer

from curverag.prompts import PROMPTS
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
        spacy_model: str= "en_core_web_lg",
    ):  
        if openai_client is None and (llm is None or outlines_llm is None):
            raise ValueError("Either an open_ai_client must be provided or both llm and outlines_llm must be provided")
        
        # Store model names for serialization
        self.gliner_model_name = gliner_model_name
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.spacy_model = spacy_model
        self.openai_model = openai_model
        
        self.using_openai = False
        if openai_client is not None:
            self.openai_client = openai_client
            self.using_openai = True
        elif llm is not None and outlines_llm is not None:
            self.llm = llm
            self.outlines_llm = outlines_llm
        else:
            raise ValueError("Both llm and outlines_llm must be provided when using a local LLM")
    
        # Initialize models
        self.gliner_model = GLiNER.from_pretrained(gliner_model_name)
        self.sentence_model = SentenceTransformer(sentence_transformer_model_name)
        self.spacy_nlp = spacy.load(spacy_model)
        
        self.entity_types = entity_types
        self.graph = None
        self.graph_embedding_model = None
        self.dataset = None
        self.node_sentence_embeddings = None
        self.edge_sentence_embeddings = None
        
    def save(self, path: str) -> None:
        """
        Save the CurveRAG instance to a file.
        
        Args:
            path: Path to save the model. If None, uses self.save_path
        """
            
        # Create a copy of the instance's dict
        state = self.__dict__.copy()
        
        # Remove non-serializable objects
        state['gliner_model'] = None
        state['sentence_model'] = None
        state['spacy_nlp'] = None
        if hasattr(self, 'openai_client'):
            state['openai_client'] = None
        if hasattr(self, 'llm'):
            state['llm'] = None
            state['outlines_llm'] = None
        
        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the state
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(
        cls,
        path: str,
        openai_client=None,
        llm=None,
        outlines_llm=None
    ) -> 'CurveRAG':
        """
        Load a saved CurveRAG instance.
        
        Args:
            path: Path to the saved model
            openai_client: OpenAI client (if using OpenAI)
            llm: Local LLM instance (if using local LLM)
            outlines_llm: Outlines LLM instance (if using local LLM)
            
        Returns:
            Loaded CurveRAG instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            openai_client=openai_client,
            openai_model=state.get('openai_model', 'gpt-4o-mini'),
            llm=llm,
            outlines_llm=outlines_llm,
            entity_types=state.get('entity_types', DEFAULT_ENTITY_TYPES),
            gliner_model_name=state.get('gliner_model_name', DEFAULT_GLINER_MODEL),
            sentence_transformer_model_name=state.get('sentence_transformer_model_name', DEFAULT_SENTENCE_TRANSFORMER_MODEL),
            spacy_model=state.get('spacy_model', 'en_core_web_lg'),
        )
        
        # Restore the state
        for key, value in state.items():
            if key not in ['gliner_model', 'sentence_model', 'spacy_nlp', 'openai_client', 'llm', 'outlines_llm']:
                setattr(instance, key, value)
        
        return instance
    
    def generate_node_embeddings(self):
        self.node_sentence_embeddings = self.sentence_model.encode([n.name for n in self.graph.nodes])
        self.edge_sentence_embeddings = self.sentence_model.encode([self.get_edge_description(self.graph, e) for e in self.graph.edges])

    def fit(self, docs: List[str], dataset_name: str):
        """Training of RAGQuery model

        And Mr RAGQuery said: Thou shalt learn the laws of the vocaublary, learn the words and their relation.
        """
        # create graph
        if self.using_openai:
            self.graph = create_graph_openai(self.openai_client, docs, model=self.openai_model)
        else:
            self.graph = create_graph_outlines(self.outlines_llm, docs)
        print('num nodes', len(self.graph.nodes))
        print('unique node ids', set([n.id for n in self.graph.nodes]))
        print('creating dataset')
        self.dataset, self.graph_embedding_nodes_id_idx, self.graph_embedding_relationship_name_idx = create_graph_dataset(self.graph, dataset_name)

        # TODO: enchance graph with general knowledge that the LLM has - how can we do this?

        # create embeddings
        print('train kg embeddings')
        self.graph_embedding_model = train(self.dataset)
        print('get node sentence embeddings')
        self.node_sentence_embeddings = self.sentence_model.encode([n.name for n in self.graph.nodes])
        print('get edge sentence embeddings')
        self.edge_sentence_embeddings = self.sentence_model.encode([self.get_edge_description(self.graph, e) for e in self.graph.edges])

    def get_query_entities(self, query, threshold, additional_entity_types):
        # get gliner entites
        if not additional_entity_types:
            additional_entity_types = []
        all_entity_types = self.entity_types + additional_entity_types
        entities = self.gliner_model.predict_entities(query, all_entity_types, threshold=threshold)
        entities = [e['text'] for e in entities]

        query_sp = self.spacy_nlp(query)
        pos_entities = []
        for token in query_sp:
            #print(token.text, token.pos_, token.tag_)
            if token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                pos_entities.append(token.text)
        entities += pos_entities
        print(entities)
        return entities
    
    def get_edge_description(self, graph, edge):
        return graph.get_matching_node_by_id(edge.source).name + " has a relationship with " + graph.get_matching_node_by_id(edge.target).name + " called " + edge.name + " and desribed as: " + edge.description 

    def get_edge_types(self, entities):
        #print('entites for edges', entities)
        entities = self.sentence_model.encode(entities)
        edges = self.sentence_model.encode([e.name for e in self.graph.edges])
        similarities = self.sentence_model.similarity(edges, entities)
        #print('similarities', similarities)
        threshold = 0.5
        similar_indices = [list(np.where(sim_row > threshold)[0]) for sim_row in similarities]
        similar_indices = list(set([i for s in similar_indices for i in s]))
        similar_edges = [e.name for i, e in enumerate(self.graph.edges) if i in similar_indices]
        #print('similar_edges', similar_edges)
        return similar_edges
    
    def query(self,
        query: str,
        additional_entity_types: Optional[List[str]]=None,
        threshold: float = 0.4,
        edge_threshold = 0.5,
        max_tokens: int = 100,
        traversal='hyperbolic',
        top_k: int = 10,
        query_prompt: str = 'generate_response_query_with_references'
    ):
        print('query:', query)
        entities = self.get_query_entities(query, threshold, additional_entity_types)

        # get all query nodes using sentence transformer
        query_embeddings = self.sentence_model.encode(entities + [query])
        similarities = self.sentence_model.similarity(query_embeddings, self.node_sentence_embeddings)
        similar_indices = [list(np.where(sim_row > threshold)[0]) for sim_row in similarities]
        similar_indices = list(set([i for s in similar_indices for i in s]))

        # get node ids of similar embeddings
        node_ids = [n.id for i, n in enumerate(self.graph.nodes) if i in similar_indices]
        
        # get addditonal nodes from graph edges using sentence transformer by using similarity of query entities to edges
        similarities = self.sentence_model.similarity(query_embeddings, self.edge_sentence_embeddings)
        #print('similarities', similarities)
        edge_similar_indices = [list(np.where(sim_row > edge_threshold)[0]) for sim_row in similarities]
        #print('edge_similar_indices', edge_similar_indices)
        edge_similar_indices = list(set([i for s in edge_similar_indices for i in s]))
        #print('edge_similar_indices', edge_similar_indices)
        for i in edge_similar_indices: # get edge node ids and and idx
            edge = self.graph.edges[i]
            #print('edge to add', edge)
            node_ids.append(edge.source)

            node_ids.append(edge.target)

        node_ids = list(set(node_ids))
        if len(node_ids) == 0:
            print("Found no embedding indx for entities, doing non KG-RAG result")
            return "N/A", None

        # get related nodes by trraversing through graph
        if traversal == 'hyperbolic':
            # get node indexes for graph embedding
            embedding_idx = [[self.graph_embedding_nodes_id_idx[n] for n in node_ids if n in self.graph_embedding_nodes_id_idx]]
            entity_node_embs = self.graph_embedding_model.entity.weight.data[embedding_idx]
            # get all embeddings
            all_node_embs = self.graph_embedding_model.entity.weight.data
            # traverse graph and get all other related nodes and entities
            similar_node_indexes = self.graph.traverse_hyperbolic_embeddings(entity_node_embs, all_node_embs, threshold=0.6, top_k=top_k)
            similar_node_indexes = [int(i) for s in similar_node_indexes for i in s]
            similar_node_indexes = list(set(similar_node_indexes))
            graph_embedding_nodes_idx_id = {v: k for k, v in self.graph_embedding_nodes_id_idx.items()}
            similar_node_ids = [graph_embedding_nodes_idx_id[idx] for idx in similar_node_indexes]
        elif traversal == 'pp':
            edge_types = self.get_edge_types(entities)
            similar_node_ids = self.graph.traverse_personalised_pagerank(node_ids, top_k=top_k, edge_types=edge_types)
        #print('similar_node_ids', similar_node_ids)
        #print('similar_node_ids graph nodes retrieved', [n.name for n in self.graph.nodes if n.id in similar_node_ids])
          
        # add nodes retrieved from grpah rtarversal to all nodes that will be added to prompy
        node_ids += similar_node_ids 
        node_ids = list(set(node_ids))

        # create a subgraph including all nodes we want to add to prompt                            
        query_graph = self.graph.get_subgraph(node_ids)

        # add descriptions of nodes and entities to prompt, along with query 
        prompt_args = {"query": query, "context": str(query_graph)}
        prompt = PROMPTS[query_prompt] # use query and query_graph
        prompt = prompt.format(**prompt_args)
        #print(prompt)

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

        return result, query_graph
