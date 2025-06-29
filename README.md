# Curve RAG

Novel RAG approach using a graph approach using a knowledge graph approach with novel edge weighted personalised page rank and hyperbolic embeddings.

How it works (abbreviated):
- Creates graph using an LLM
- Creates embeddings of graph nodes using [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/abs/2005.00545) [6]
- At query time finds relevant nodes and relationships from graph using both personalised page rank and hyperbolic embeddings.
- Retrieved nodes and embeddings are used as context to LLM

NOTE: Other advances in this approach will be added to the README at a later date

## Usage

To train the model and get embeddings for use for RAG use the following:
```
import llama_cpp
from curverag import utils
from curverag.curverag import CurveRAG

# Define curverag LLM and documents 
max_tokens = 10000
n_ctx=10000
docs = [
    "....",
    "....",
    ...
]
model = utils.load_model(
    llm_model_path="./models/Meta-Llama-3-8B-Instruct.Q6_K.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct"),
    n_ctx=n_ctx,
    max_tokens=max_tokens
)

# Train CurveRAG model
rag = CurveRAG(llm=model) # alternatively can ue openai
rag.fit(docs, dataset_name='dataset')

rag.query("My query...", traversal = 'pp') # rag query using personalised page rank
rag.query("My query...", traversal = 'hyperbolic') # rag approach using hyperbolic embeddings
```

## Config

Model config settins are defined in the `config.toml` file