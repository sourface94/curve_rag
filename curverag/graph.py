from pydantic import BaseModel, Field
from typing import List, Optional

from tqdm import tqdm
import llama_cpp
from outlines import generate, models
from llama_cpp import Llama

from curverag.transformations import chunk_text
from curverag.prompts import PROMPTS


class Node(BaseModel):
    id: str = Field(..., description="Unique identifier of the node")
    name: str = Field(..., description="Name of the node")
    description: str = Field(..., description="Description of the node")
    alias: List[str] = Field(..., description="Other names used to identify the node")
    #attributes: List[str] = Field(..., description="Attributes used to describe the node")

class Edge(BaseModel):
    source: str = Field(..., description="Name of the source edge")
    target: str = Field(..., description="Name of the target edge")
    name: str = Field(..., description="Name of the relationship for the edge")
    is_directed: bool  = Field(..., description="If true its a directed edge")
    description: str = Field(..., description="Description of the edge")
    #attributes: List[str] = Field(..., description="Attributes used to describe the ege")


class KnowledgeGraph(BaseModel):

    nodes: List[Node] = Field(..., description="List of nodes of the knowledge graph")
    edges: List[Edge] = Field(..., description="List of edges of the knowledge graph")

    def is_empty(self) -> bool:
        return len(self.nodes) == 0 and len(self.edges) == 0

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def get_matching_node(self, node: Node):
        for n in self.nodes:
            if node.name == n.name or node.name in n.alias:
                return n
        return

    def get_matching_edge(self, edge: Edge):
        for e in self.edges:                
            if edge.source == e.source and edge.target == e.target and edge.is_directed == e.is_directed and edge.description == e.description:
                return e
        return

    def upsert(self, sub_graph: "KnowledgeGraph"):
        if self.is_empty():
            self.nodes = sub_graph.nodes
            self.edges = sub_graph.edges
            return
    
        for node in sub_graph.nodes:
            matching_node = self.get_matching_node(node)
            if matching_node is None:
                self.add_node(node)

        for edge in sub_graph.edges:
            matching_edge = self.get_matching_edge(edge)
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
        return self.nodes[:top_k]


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


def generate_prompt(user_prompt, schema):
    return f""""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
       You are a world class AI model who extracts ndoes and entities from documents for a Knowledge Graph creation task. Put yur reply in JSON<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Here's the json schema you must adhere to: <schema>{schema}</schema><|im_end|>
        Here is the text you must extract nodes and entities for:
        {user_prompt}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def create_graph(model, texts: List[str], is_narrative: bool = False, max_tokens=1000, chunk_size: int = 1028):
    """
    Create knowledge graph.
    """
    
    generator = generate.json(model, KnowledgeGraph)
    # load graph schema and empty graph
    schema = KnowledgeGraph.model_json_schema()
    graph = KnowledgeGraph(nodes=[], edges=[])

    # chunk text
    texts = chunk_text(texts, chunk_size)

    for chunk in tqdm(texts):
        prompt = generate_prompt(chunk, schema)
        print(prompt)
        sub_graph = generator(prompt, max_tokens=max_tokens, temperature=0, seed=42)
        print('sub graph', sub_graph)
        print('----------------------------------------------------------')
        graph.upsert(sub_graph)
        print('graph', graph)
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')

    return graph


def learn_embeddings(graph: KnowledgeGraph):
    return NotImplementedError





