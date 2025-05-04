# Curve RAG

Rag approach using hyperbolic geometry knowledge graphs.

Steps:
- Creates graph using an LLM
- Creates embeddings of graph nodes using [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/abs/2005.00545) [6]
- Allows RAG queries, using graph embeddings to find relevant information

## Usage (To update)

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: python .\run.py --multi_c --debug --max_epoch 3 --dataset dataset_name
```

## Config

Valid config values:
```
dataset: str = "medical_docs"
init_size: float = 1e-3
learning_rate: float = 1e-1
bias: str = "constant" #"learn"
rank: int = 1000 # embedding dimensions
gamma: float = 0
data_type: str = "double"
dtype: str = "double"
debug: bool = True
multi_c: bool = True
double_neg: bool True
neg_sample_size: int = 50
dropout: float = 0
max_epochs: int = 50
valid: float = 3 # number of epochs before validation
pateince: int = 10 # Number of epochs before early stopping
batch_size: int = 1000
optimizer: str = "Adagrad"  #"Adagrad", "Adam", "SparseAdam"
regularizer: str = "N3" #F2
reg: float = 0 # regularisation weight
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10", "large_dummy_data", "medical_docs"],
```

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

