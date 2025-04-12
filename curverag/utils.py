from typing import List, Optional

from outlines import generate, models
from llama_cpp import Llama


def load_model(llm_model_path="./models/7B/llama-model.gguf", tokenizer: str = Optional[None], n_ctx: int = Optional[None]):
    # load model
    llm = Llama(
      llm_model_path=llm_model_path,
      tokenizer=tokenizer,
      n_ctx=n_ctx,
      # n_gpu_layers=-1,
      # seed=1337,
    )
    model = models.LlamaCpp(llm)
    return model