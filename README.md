# Curve RAG

Rag approach using hyperbolic geometry knowledge graphs


# Hyperbolic Knowledge Graph Embedding 

This code is the official PyTorch implementation of [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/abs/2005.00545) [6] as well as multiple state-of-the-art KG embedding models which can be trained for the link prediction task. A Tensorflow implementation is also available at: [https://github.com/tensorflow/neural-structured-learning/tree/master/research/kg_hyp_emb](https://github.com/tensorflow/neural-structured-learning/tree/master/research/kg_hyp_emb)


## Usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
useage: python .\run.py --multi_c --debug --max_epoch 3
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}
                        Knowledge Graph dataset
  --model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
```
## Citation

If you use the codes, please cite the following paper [6]:

```
@inproceedings{chami2020low,
  title={Low-Dimensional Hyperbolic Knowledge Graph Embeddings},
  author={Chami, Ines and Wolf, Adva and Juan, Da-Cheng and Sala, Frederic and Ravi, Sujith and R{\'e}, Christopher},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={6901--6914},
  year={2020}
}
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

Some of the code was forked from the original ComplEx-N3 implementation which can be found at: [https://github.com/facebookresearch/kbc](https://github.com/facebookresearch/kbc)

