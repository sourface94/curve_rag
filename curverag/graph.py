from pydantic import BaseModel, Field
from typing import List, Optional

import torch
import llama_cpp
import numpy as np
from tqdm import tqdm
from outlines import generate, models
from llama_cpp import Llama

from curverag.transformations import chunk_text
from curverag.prompts import PROMPTS
from curverag.utils import save_kg_dataset, split_triples, generate_to_skip, create_atth_dataset
from curverag.atth.kg_dataset import KGDataset
from curverag.atth.utils.hyperbolic import hyp_distance


class Node(BaseModel):
    id: int = Field(..., description="Unique identifier of the node")
    name: str = Field(..., description="Name of the node. Maximum of 20 characters.")
    description: str = Field(..., description="A short description of the node. Maximum of 50 characters.")
    alias: List[str] = Field(..., description="Other names used to identify the node")
    additional_information: List[str] = Field(..., description="A list of additional pieces of information about the node that further describes it", max_length=5)

class Edge(BaseModel):
    source: int = Field(..., description="ID of the source node edge")
    target: int = Field(..., description="ID of the target node edge")
    name: str = Field(..., description="Name of the relationship for the edge. Maximum of 20 characters.")
    is_directed: bool  = Field(..., description="If true its a directed edge")
    description: str = Field(..., description="A short description of the edge. Maximum of 50 characters.")
    notes: List[str] = Field(..., description="A list of additional pieces of information about the edge that further describes it", max_length=5)


class KnowledgeGraph(BaseModel):

    nodes: List[Node] = Field(..., description="List of nodes of the knowledge graph. Maximum of 10 items in this list.", max_length=10)
    edges: List[Edge] = Field(..., description="List of edges of the knowledge graph. Maximum of 10 items in this list.", max_length=10)


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

    def get_matching_node_by_id(self, node_id: int):
        """Find a node with the given ID in the graph."""
        for n in self.nodes:
            if node_id == n.id:
                return n
        return None

    def get_next_node_id(self):
        """Return the next available node ID."""
        if not self.nodes:
            return 0
        return max(node.id for node in self.nodes) + 1

    def get_matching_edge(self, edge: Edge):
        for e in self.edges:                
            if edge.source == e.source and edge.target == e.target and edge.is_directed == e.is_directed and edge.description == e.description:
                return e
        return

    def upsert(self, sub_graph: "KnowledgeGraph"):
        print('upserting')
        if self.is_empty():
            self.nodes = sub_graph.nodes
            self.edges = sub_graph.edges
            return

        # Create a mapping of original subgraph node IDs to potentially new IDs
        id_mapping = {}
        
        # First pass: process nodes and build ID mapping
        for node in sub_graph.nodes:
            # Check if a semantically matching node exists
            matching_node = self.get_matching_node(node)
            
            if matching_node is not None:
                # If we found a semantic match, map the subgraph node ID to the existing node ID
                id_mapping[node.id] = matching_node.id
            else:
                # Check if the node ID already exists in the graph
                existing_node_with_same_id = self.get_matching_node_by_id(node.id)
                
                if existing_node_with_same_id is not None:
                    # ID conflict: assign a new ID to this node
                    new_id = self.get_next_node_id()
                    id_mapping[node.id] = new_id
                    
                    # Create a new node with the updated ID
                    new_node = Node(
                        id=new_id,
                        name=node.name,
                        description=node.description,
                        alias=node.alias,
                        additional_information=node.additional_information
                    )
                    self.add_node(new_node)
                else:
                    # No conflict, add the node as is
                    id_mapping[node.id] = node.id
                    self.add_node(node)
        
        # Second pass: process edges with updated node IDs
        for edge in sub_graph.edges:
            # Update source and target IDs based on the mapping
            source_id = id_mapping.get(edge.source, edge.source)
            target_id = id_mapping.get(edge.target, edge.target)
            
            # Create a new edge with updated IDs
            updated_edge = Edge(
                source=source_id,
                target=target_id,
                name=edge.name,
                is_directed=edge.is_directed,
                description=edge.description,
                notes=edge.notes
            )
            
            # Check if this edge already exists
            matching_edge = self.get_matching_edge(updated_edge)
            if matching_edge is None:
                self.add_edge(updated_edge)

    def traverse(self, query: str):
        """Traverse using custom traverse algorithm"""
        
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


    def traverse_hyperbolic_embeddings(self, node_embeddings: torch.Tensor, all_embeddings: torch.Tensor, top_k: int=3, threshold: float=0.7, curvature: float=1.0):
        """Traverse using hyperbolic embeddings"""
        
        # Get curvature value c (assume c=1.0 here, or retrieve from model/config if available)
        c = torch.tensor([curvature], dtype=node_embeddings.dtype, device=node_embeddings.device)

        all_distances = hyp_distance(node_embeddings, all_embeddings, c, eval_mode=True)

        node_nn_ids = []
        for distances in all_distances:
            scores, indices = torch.topk(distances, top_k, largest=False)
            mask = scores > threshold
            filtered_vals = scores[mask]
            filtered_indices = indices[mask]
            node_nn_ids.append(filtered_indices)

        return node_nn_ids

    def get_subgraph(self, node_ids: List[int]):
        nodes = [n for n in self.nodes if n.id in node_ids]
        edges = []
        for e in self.edges:
            if e.source in node_ids and e.target in node_ids:
                edges.append(e)
        return KnowledgeGraph(nodes=nodes, edges=edges)


    def __str__(self):
        lines = []
        lines.append("KnowledgeGraph Overview")
        lines.append(f"  There are {len(self.nodes)} entities and {len(self.edges)} relationships in this graph.\n")

        # Nodes section
        lines.append("Entities in the graph:")
        for node in self.nodes:
            lines.append(f"  • '{node.name}':")
            lines.append(f"      The entity has the following description: {node.description}")
            if node.alias:
                lines.append(f"      It can also be referred to as: {', '.join(node.alias)}")
            print('***************************************!!******************************')
            if node.alias:
                lines.append(f"      It has the following additional information: {', '.join(node.additional_information)}")

        # Relationships section
        lines.append("\nRelationships between nodes:")
        seen = set()
        node_id_to_name = {n.id: n.name for n in self.nodes}
        for edge in self.edges:
            # For undirected, sort ids to avoid repeats
            if not edge.is_directed:
                key = tuple(sorted([edge.source, edge.target])) + (edge.name,)
            else:
                key = (edge.source, edge.target, edge.name)
            if key in seen:
                continue
            seen.add(key)
            src_name = node_id_to_name.get(edge.source, str(edge.source))
            tgt_name = node_id_to_name.get(edge.target, str(edge.target))
            if edge.is_directed:
                lines.append(
                    f"  • There is a directed relationship from '{src_name}' to '{tgt_name}' called '{edge.name}'."
                )
            else:
                lines.append(
                    f"  • There is an undirected relationship between '{src_name}' and '{tgt_name}' called '{edge.name}'."
                )
            lines.append(f"      The relationship is described as: {edge.description}")
            lines.append(f"      The relationship has the following notes: {', '.join(edge.notes)}")

        return "\n".join(lines)


def generate_prompt(user_prompt, schema, existing_graph):
    return f""""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
       You are a world class AI model who extracts nodes and entities from documents to add to an exiting Knowledge Graph creation task. Put yur reply in JSON<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Here's the json schema you must adhere to: <schema>{schema}</schema><|im_end|>
        Here is the knowledge graph that you will be adding to: {existing_graph}
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
        prompt = generate_prompt(chunk, schema, graph.json())
        sub_graph = generator(prompt, max_tokens=max_tokens, temperature=0, seed=42)
        graph.upsert(sub_graph)

    return graph

def create_graph_dataset(graph: KnowledgeGraph, dataset_name: str):
    triples, nodes_id_idx, relationship_name_idx = create_atth_dataset(graph)
    train, valid, test = split_triples(triples)
    all_triples = np.concatenate([train, valid, test], axis=0)
    to_skip = generate_to_skip(all_triples)
    
    save_kg_dataset(all_triples, nodes_id_idx, relationship_name_idx, "./data/" + dataset_name)
    dataset = KGDataset("./data/" + dataset_name, debug=False, name=dataset_name)

    return dataset







