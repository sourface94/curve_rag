from typing import List, Optional

from outlines import generate, models
from llama_cpp import Llama

from curverag.graph import KnowledgeGraph


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
    model = models.LlamaCpp(llm)
    return model


def create_atth_dataset(graph: KnowledgeGraph):
  nodes_idx = {n: id for i, n.name in enumerate(graph.nodes)}
  edges_ = set([e.name for e in graph.edges])
  edges = {e for i, e for e in enumerate(edges_)}
  relationship_types = {n: id for i, n.name in enumerate(graph.nodes)}
  
