import torch.nn.functional as F

from torch import nn

from curverag.hyperbolic_utils import hyperbolic_distance


class AttH():

    def __init__(
        
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        init_size: float = 0.001, # float for embeddings' initialization scale
        data_type = torch.float

    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # create embeddings for nodes, relations and bias'
        self.entity_emb = nn.Embedding(num_entities, self.embedding_dim)
        self.entity.weight.data = self.init_size * torch.randn((self.num_entities, self.embedding_dim), dtype=self.data_type)
        self.rel_emb = nn.Embedding(num_relations, self.embedding_dim)
        self.rel.weight.data = self.init_size * torch.randn((self.num_relations, 2 * self.embedding_dim), dtype=self.data_type)

        self.bias_head = nn.Embedding(self.num_entities, 1)
        self.bias_head.weight.data = torch.zeros((self.num_entities, 1), dtype=self.data_type)
        self.bias_tail = nn.Embedding(self.num_entities, 1)
        self.bias_tail.weight.data = torch.zeros((self.num_entities, 1), dtype=self.data_type)

        # creare curvature
        c_init = torch.ones((self.num_relations, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

        # create attention embeddings and initialise values
        self.att_rel_emb = nn.Embedding(self.num_relations, 2 * self.embedding_dim)
        self.att_ rel_emb.weight.data = 2 * torch.rand((self.num_relations, 2 * self.embedding_dim), dtype=self.data_type) - 1.0
        self.context_emb = nn.Embedding(self.num_relations, self.embedding_dim)
        self.context_emb.weight.data = self.init_size * torch.randn((self.num_relations, self.embedding_dim), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)

        # used to scale values
        if data_type == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.embedding_dim)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.embedding_dim)]).cuda()
    

    def forward(self, queries, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and biases for query
        lhs_embs, lhs_biases, curvatures = self.get_queries(queries)

        # get embeddings and biases for targets
        rhs_embs, rhs_biases = self.get_rhs(queries, eval_mode)
        
        # get predictions
        predictions = self.score(lhs_e, lhs_biases, curvatures, rhs_e, rhs_biases, eval_mode)

        # get factors for regularization
        factors = self.get_factors(queries)
        return predictions, factors

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return res, self.bh(queries[:, 0]), c

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2]) # getting index which is tail entity in (h, r, t) triple


    def score(self, lhs_embs, lhs_bias rhs_embs, c, rhs_bias, eval_mode):
        """Get similarity scores or queries against targets in embedding space."""
        lhs_e, c 
        distance = hyperbolic_distance(lhs_e, rhs_e, c, eval_mode) ** 2
        if eval_mode:
            return lhs_biases + rhs_biases.t() + score
        else:
            return lhs_bias + rhs_bias + distance