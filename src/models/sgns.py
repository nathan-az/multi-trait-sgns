from typing import List, Dict

import torch
from torch import nn


class MultiEmbeddingSGNS(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, side_info_specs: Dict[str, int]
    ):
        super().__init__()

        self.num_embeddings = 1 + len(side_info_specs)
        self.embedding_dim = embedding_dim

        self.target_embeddings = nn.ModuleDict(
            {
                "base": nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                ),
                **{
                    name: nn.Embedding(
                        num_embeddings=size,
                        embedding_dim=embedding_dim,
                    )
                    for name, size in side_info_specs.items()
                },
            }
        )
        self.context_embeddings = nn.ModuleDict(
            {
                "base": nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                ),
                **{
                    name: nn.Embedding(
                        num_embeddings=size,
                        embedding_dim=embedding_dim,
                    )
                    for name, size in side_info_specs.items()
                },
            }
        )

        self.target_embedding_weights = nn.Parameter(
            torch.empty(self.num_embeddings, 1)
        )
        nn.init.xavier_normal(self.target_embedding_weights)

        self.context_embedding_weights = nn.Parameter(
            torch.empty(self.num_embeddings, 1)
        )
        nn.init.xavier_normal(self.context_embedding_weights)
        self.weight_softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def indices_to_embedding_vectors(self, x, embeddings: List[nn.Module]):
        """
        Returns a tensor of vectors for each batch, for each embedding

        :param x: Tensor of indices with shape (batch_size, num_embeddings)
        :param embeddings: List of nn.Embedding modules, where len(embeddings) == x.shape[1]
        :return:
        Tensor of shape (bs, num_embeddings, embedding_dim) with indices turned to their corresponding vectors.
        """
        # x should be a tensor of shape (batch_size, self.num_embeddings)
        bs, num_embeddings = x.shape

        out = torch.stack(
            [
                embedding(x[b, i])
                for b in range(bs)
                for i, embedding in enumerate(embeddings)
            ]
        )  # bs * num_embeddings, dim
        out = out.reshape(
            bs, num_embeddings, self.embedding_dim
        )  # bs, num_embeddings, dim
        return out

    def _weight_embeddings(self, embedding_batch, embedding_weights):
        normalised_weights = self.weight_softmax(embedding_weights)
        out = torch.einsum(
            "bnd,nl->bld", embedding_batch, normalised_weights
        )  # bs, batch_size, 1
        return out

    def forward_target(self, x):
        """For extracting the weighted target embedding, x specifies indices for each type of info"""
        embedding_batch = self.indices_to_embedding_vectors(
            x, embeddings=[embedding for embedding in self.target_embeddings.values()]
        )
        out = self._weight_embeddings(embedding_batch, self.target_embedding_weights)
        return out

    def forward_context(self, x):
        """For extracting the weighted context embedding, x specifies indices for each type of info"""
        embedding_batch = self.indices_to_embedding_vectors(
            x, embeddings=[embedding for embedding in self.context_embeddings.values()]
        )
        out = self._weight_embeddings(embedding_batch, self.context_embedding_weights)
        return out

    def forward(self, x):
        # x.shape (bs, 2, num_embeddings) should contain indices
        target_indices = x[:, 0, :]
        context_indices = x[:, 0, :]

        target_embedding = self.forward_target(target_indices)
        context_embedding = self.forward_context(context_indices)

        logits = torch.matmul(target_embedding, context_embedding.transpose(1, 2))
        out = self.sigmoid(logits).squeeze()  # (bs, 1) - can fully squeeze if needed
        return out

    def extract_mean_embedding(self, x):
        target_embedding = self.forward_target(x)
        context_embedding = self.forward_context(x)
        return (target_embedding + context_embedding) / 2
