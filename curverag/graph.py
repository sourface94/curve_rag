from dataclasses import dataclass
from typing import List

from llama_cpp import Llama

from curverag.transformations import chunk_text
from curverag.prompts import entity_relationship_extraction_disparate_prompt


@dataclass
class Node:
    id: str
    description: str
    alias: List[str]
    attributes: List[str]


class Edge:
    node_1_id: str
    node_2_id: str
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
        return NotImplementedError

    def get_matching_edge(self, edge: Edge):
        return NotImplementedError

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


def create_graph(texts: List[str], is_narrative: bool = False, llm_model_path="./models/7B/llama-model.gguf", max_tokens=1000):
    """
    Create knowledge graph.

    Creation of this packages knowledge center
    """
    
    # chunk text
    texts = chunk_text(texts)

    # load model
    llm = Llama(
      llm_model_path="./models/7B/llama-model.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
    )

    graph = Graph()
    for chunk in texts:
        sub_graph = llm(
            entity_relationship_extraction_disparate_prompt + chunk, # Prompt
            max_tokens=max_tokens,
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
        graph.upsert(sub_graph)


def learn_embeddings():
    return NotImplementedError

