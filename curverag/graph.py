from typing import List

from llama_cpp import Llama

from curverag.transformations import chunk_text
from curverag.prompts import entity_relationship_extraction_disparate_prompt





class Graph:
    def __init__():
        pass


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

    for chunk in texts:
        output = llm(
            entity_relationship_extraction_disparate_prompt + chunk, # Prompt
            max_tokens=max_tokens,
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
        print(output)





