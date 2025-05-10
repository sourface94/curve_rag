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

    nodes_id_idx = {n.id: i for i, n in enumerate(graph.nodes)}
    nodes_idx_id = {idx: id for id, idx in nodes_id_idx.items()}

    edges_ = set([e.name for e in graph.edges])
    edges = {e for i, e in enumerate(edges_)}
    relationship_name_idx = {n: i for i, n in enumerate(edges_)}
    relationship_idx_name = {idx: name for name, idx in relationship_name_idx.items()}

    graph_list = []
    for e in graph.edges:
        graph_list.append([nodes_id_idx[e.source], relationship_name_idx[e.name], nodes_id_idx[e.target]])

    return np.array(graph_list), nodes_id_idx, relationship_name_idx


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
