from dataclasses import dataclass
from typing import List

from llama_cpp import Llama

from curverag.transformations import chunk_text
from curverag.prompts import PROMPTS


@dataclass
class Node:
    id: str
    description: str
    alias: List[str]
    attributes: List[str]

    def __str__():



class Edge:
    node_1_id: str
    node_2_id: str
    name: str
    direction: bool
    description: str
    attributes: List[str]


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def get_matching_node(self, node: Node):
        for n in self.nodes:
            if n.id == node.id:
                return n
        return

    def get_matching_edge(self, edge: Edge):
        for e in self.edges:
            if e.node_1_id == edge.node_1_id and e.node_2_id == edge.node_2_id and  e.name == edge.name and e.direction == edge.direction:
                return e
        return

    def upsert(self, sub_graph: Graph):
        if self.is_empty():
            self.nodes = sub_graph.nodes
            self.relationships = sub_graph.relationships
            return
    
        for node in sub_graph.nodes:
            matching_node = self.get_matching_node()
            if matching_node is None:
                self.add_node(node)

        for edge in sub_graph.edges:
            matching_edge = self.get_matching_edge()
            if matching_edge:
                self.add_edge(edge)

    def traverse(self, query: str):
        """Traverse using custom traverse algorithm"""
        # get each entity in the query using gliner
        entities = get_entities(query)
        
        # look for entities as nodes in the graph
        entities_in_graph = get_nodes_by_name(entities)

        # get the importance of each entity
        scores = score_entities(query, entities_in_graph)

        # traverse the graph, giving extra weight to nodes which connect (1 hop, 2 hop, 3 hop etc) two entites together (do this by maintaining a register of visited nodes) and 
        # weight to edges which have the same name/description as entities not matched as nodes). Traverse for longer for entities with higher scores. Return all nodes that were traversed
        nodes = self.weighted_traverse(entities, entities_in_graph)

        return nodes


    def traverse_personalised_pagerank(self, query: str, top_k: int):
        """Traverse using personalised pagerank algorithm
        Use either igraph or networkx implementation
        """
        # get each entity in the query using gliner
        entities = get_entities(query)
        
        # look for entities as nodes in the graph
        entities_in_graph = get_nodes_by_name(entities)

        # get the importance of each node
        node_ranks = page_rank(...)

        # get the top k nodes
        return nodes[:top_k]


    def traverse_hyperbolic_embeddings(self, query: str, top_k: int):
        """Traverse using hyperbolic embeddings"""
        
        # get each entity in the query using gliner
        entities = get_entities(query)
        
        # look for entities as nodes in the graph
        entities_in_graph = get_nodes_by_name(entities)

        # get embeddings
        embeddings = node_embeddings()
        
        # get nerest neighburs using embeddings
        nearest_nodes = nn(embeddings, entities_in_graph)
        
        # get nearest neigbour in a 1 and 2 hop fashion when the neighbours join two entites e.g. is a neighboiur of a neighbour
        nearest_hop_nodes = nn_hop(embeddings, entities_in_graph, hop=2)


        # get the top k nodes
        return nearest_nodes + nearest_hop_nodes


def create_graph(texts: List[str], is_narrative: bool = False, llm_model_path="./models/7B/llama-model.gguf", max_tokens=1000):
    """
    Create knowledge graph.

    Creation of this packages knowledge center
    """
    
    # chunk text
    texts = chunk_text(texts)

    # load model
    llm = Llama(
      llm_model_path=llm_model_path,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
    )

    graph = Graph()
    for chunk in texts:
        sub_graph = llm(
            # TODO: fix this prompt, use outlines
            PROMPTS["entity_relationship_extraction_disparate_prompt"] + chunk, # Prompt
            max_tokens=max_tokens,
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
        graph.upsert(sub_graph)

    return graph


def learn_embeddings(graph: Graph):
    return NotImplementedError





