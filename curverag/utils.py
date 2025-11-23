import pickle
import os
from typing import List, Optional

import numpy as np
from outlines import generate, models
from llama_cpp import Llama


def load_model(
        llm_model_path: str ="./models/7B/llama-model.gguf",
        tokenizer: str = Optional[None],
        max_tokens: int = 100,
        n_ctx: int = 2000
    ):
    """Load LLM model using llama.cpp and outlines"""
    # load model using llama.cpp
    llm = Llama(
      model_path=llm_model_path,
      tokenizer=tokenizer,
      max_tokens=max_tokens,
      n_ctx=n_ctx,
      # n_gpu_layers=-1,
      # seed=1337,
    )
    # create outlines model
    outlines_model = models.LlamaCpp(llm)
    return llm, outlines_model


def create_atth_dataset(graph):
    # Get all unique nodes and relationships
    all_nodes = set()
    relationships = set()
    
    for e in graph.edges:
        all_nodes.add(e.source)
        all_nodes.add(e.target)
        relationships.add(e.name)
    
    # Create zero-based contiguous indices
    nodes_id_idx = {node: i for i, node in enumerate(sorted(all_nodes))}
    relationship_name_idx = {rel: i for i, rel in enumerate(sorted(relationships))}
    
    # Create graph list with mapped indices
    graph_list = []
    for e in graph.edges:
        graph_list.append([
            nodes_id_idx[e.source],
            relationship_name_idx[e.name],
            nodes_id_idx[e.target]
        ])
    
    return np.array(graph_list), nodes_id_idx, relationship_name_idx


"""
def split_triples(triples, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    idxs = np.random.permutation(len(triples))
    n_total = len(triples)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_idx = idxs[:n_train]
    valid_idx = idxs[n_train:n_train + n_valid]
    test_idx = idxs[n_train + n_valid:]
    
    return triples[train_idx], triples[valid_idx], triples[test_idx]
"""

def split_triples(triples, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    
    # Identify all unique nodes (head and tail entities)
    all_nodes = set()
    for h, r, t in triples:
        all_nodes.add(int(h))
        all_nodes.add(int(t))
    
    # Create a dictionary to track which nodes are covered
    node_covered = {node: False for node in all_nodes}
    
    # Shuffle the triples
    shuffled_indices = np.random.permutation(len(triples))
    shuffled_triples = triples[shuffled_indices]
    
    # First pass: select triples to ensure all nodes are in the training set
    essential_indices = []
    remaining_indices = []
    
    for i, idx in enumerate(shuffled_indices):
        h, r, t = triples[idx]
        h, t = int(h), int(t)
        
        # If either node is not yet covered, add this triple to essential set
        if not node_covered[h] or not node_covered[t]:
            essential_indices.append(idx)
            node_covered[h] = True
            node_covered[t] = True
        else:
            remaining_indices.append(idx)
    
    # Verify all nodes are covered
    assert all(node_covered.values()), "Failed to cover all nodes in training set"
    
    # Calculate how many additional triples we need for each set
    n_essential = len(essential_indices)
    n_remaining = len(remaining_indices)
    
    # Adjust ratios for remaining triples
    remaining_train_ratio = max(0, (train_ratio * len(triples) - n_essential) / n_remaining)
    remaining_valid_ratio = valid_ratio / (valid_ratio + test_ratio) * (1 - remaining_train_ratio)
    remaining_test_ratio = 1 - remaining_train_ratio - remaining_valid_ratio
    
    # Split remaining triples
    n_remaining_train = int(n_remaining * remaining_train_ratio)
    n_remaining_valid = int(n_remaining * remaining_valid_ratio)
    
    # Combine essential triples with additional training triples
    train_indices = essential_indices + remaining_indices[:n_remaining_train]
    valid_indices = remaining_indices[n_remaining_train:n_remaining_train + n_remaining_valid]
    test_indices = remaining_indices[n_remaining_train + n_remaining_valid:]
    
    return triples[train_indices], triples[valid_indices], triples[test_indices]

def generate_to_skip(triples):
    # triples: np.ndarray of shape [N, 3]
    
    to_skip = {'lhs': {}, 'rhs': {}}
    for h, r, t in triples:
        h = int(h)
        r = int(r)
        t = int(t)
        # For filtered ranking on the right (predict tail)
        if (h, r) not in to_skip['rhs']:
            to_skip['rhs'][(h, r)] = set()
        to_skip['rhs'][(h, r)].add(t)
        # For filtered ranking on the left (predict head)
        if (t, r) not in to_skip['lhs']:
            to_skip['lhs'][(t, r)] = set()
        to_skip['lhs'][(t, r)].add(h)
            
    # Convert sets to sorted lists for serialization
    for side in ['lhs', 'rhs']:
        for k in to_skip[side]:
            to_skip[side][k] = sorted(list(to_skip[side][k]))
    return to_skip


def save_kg_dataset(triples, nodes_id_idx, relationship_name_idx, output_dir):
    train, valid, test = split_triples(triples)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pickle"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(output_dir, "valid.pickle"), "wb") as f:
        pickle.dump(valid, f)
    with open(os.path.join(output_dir, "test.pickle"), "wb") as f:
        pickle.dump(test, f)
    # Combine all triples for to_skip
    all_triples = np.concatenate([train, valid, test], axis=0)
    to_skip = generate_to_skip(all_triples)
    with open(os.path.join(output_dir, "to_skip.pickle"), "wb") as f:
        pickle.dump(to_skip, f)

    # save name to idx files
    with open(os.path.join(output_dir, "nodes_id_idx.pickle"), "wb") as f:
        pickle.dump(nodes_id_idx, f)
    with open(os.path.join(output_dir, "relationship_name_idx.pickle"), "wb") as f:
        pickle.dump(relationship_name_idx, f)


def get_gliner_entities(text, gliner_model, threshold: float = 0.4, additional_entity_types: Optional[List[str]] = None):
    # get gliner entites
    if not additional_entity_types:
        additional_entity_types = []
    entity_types = ["person", "location", "entity", "organisation"]

    all_entity_types = entity_types + additional_entity_types
    entities = gliner_model.predict_entities(text, all_entity_types, threshold=threshold)
    return entities

def get_edges(sentence_model, entities, graph):
    #print('entites for edges', entities)
    entities = sentence_model.encode(entities)
    edges = sentence_model.encode([e.name for e in graph.edges])
    similarities = sentence_model.similarity(edges, entities)
    #print('similarities', similarities)
    threshold = 0.5
    similar_indices = [list(np.where(sim_row > threshold)[0]) for sim_row in similarities]
    similar_indices = list(set([i for s in similar_indices for i in s]))
    similar_edges = [e.name for i, e in enumerate(graph.edges) if i in similar_indices]
    #print('similar_edges', similar_edges)
    return similar_edges

def get_edge_description(graph, edge):
    return graph.get_matching_node_by_id(edge.source).name + " has a relationship with " + graph.get_matching_node_by_id(edge.target).name + " called " + edge.name + " and desribed as: " + edge.description 
