# Curve RAG

Rag approach using hyperbolic geometry knowledge graphs.

How it works:
- Creates graph using an LLM
- Creates embeddings of graph nodes using [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/abs/2005.00545) [6]
- Allows RAG queries, using graph embeddings to find relevant information to use for RAG.

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
rag = CurveRAG(llm=model)
rag.fit(docs, dataset_name='test_run')

# Get node ID to embedding index mapping
nodes_id_idx = rag.dataset.nodes_id_idx

# Get node embeddings for node with ID 1
node_embs = rag.model.entity.weight.data.cpu().numpy()
node_embs[nodes_id_idx[1]]
```

## Config

Model config settins are defined in the `config.toml` file

## References

[1] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
International Conference on Machine Learning. 2016.

[2] Lacroix, Timothee, et al. "Canonical Tensor Decomposition for Knowledge Base
Completion." International Conference on Machine Learning. 2018.

[3] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational
rotation in complex space." International Conference on Learning
Representations. 2019.

[4] Bordes, Antoine, et al. "Translating embeddings for modeling
multi-relational data." Advances in neural information processing systems. 2013.

[5] Balažević, Ivana, et al. "Multi-relational Poincaré Graph Embeddings."
Advances in neural information processing systems. 2019.

[6] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings."
Annual Meeting of the Association for Computational Linguistics. 2020.

```
@inproceedings{chami2020low,
  title={Low-Dimensional Hyperbolic Knowledge Graph Embeddings},
  author={Chami, Ines and Wolf, Adva and Juan, Da-Cheng and Sala, Frederic and Ravi, Sujith and R{\'e}, Christopher},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={6901--6914},
  year={2020}
}
```

Some of the code was forked from the original ComplEx-N3 implementation which can be found at: [https://github.com/facebookresearch/kbc](https://github.com/facebookresearch/kbc)

