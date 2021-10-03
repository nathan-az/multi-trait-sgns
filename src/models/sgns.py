from typing import List, Dict

import torch
from torch import nn


class MultiEmbeddingSGNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, side_info_specs: Dict[str, int]):
        super().__init__()

        self.num_embeddings = 1 + len(side_info_specs)
        self.embedding_dim = embedding_dim

        self.target_embeddings = nn.ModuleDict(
            {
                "base": nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                ),
                **{
                    name: nn.Embedding(
                        num_embeddings=size,
                        embedding_dim=embedding_dim,
                        padding_idx=0,
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
                    padding_idx=0,
                ),
                **{
                    name: nn.Embedding(
                        num_embeddings=size,
                        embedding_dim=embedding_dim,
                        padding_idx=0,
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

    def _get_embedding_batch(self, x, embeddings: List[nn.Module]):
        """For extracting the weighted target embedding"""
        # x should be a tensor of shape (batch_size, self.num_embeddings)
        bs, num_embeddings = x.shape

        out = torch.stack(
            [
                embedder(x[b, i])
                for b in range(bs)
                for i, embedder in enumerate(embeddings)
            ]
        )  # bs * num_embeddings, dim
        out = out.reshape(
            bs, num_embeddings, self.embedding_dim
        )  # bs, num_embeddings, dim
        return out

    def _weight_embeddings(self, embedding_batch, embedding_weights):
        normalised_weights = self.weight_softmax(embedding_weights)
        out = torch.einsum(
            "bnd,nl->bdl", embedding_batch, normalised_weights
        )  # bs, batch_size, 1
        return out

    def forward_target(self, x):
        """For extracting the weighted target embedding"""
        embedding_batch = self._get_embedding_batch(
            x, embeddings=[embedding for embedding in self.target_embeddings.values()]
        )
        out = self._weight_embeddings(embedding_batch, self.target_embedding_weights)
        return out

    def forward_context(self, x):
        """For extracting the weighted target embedding"""
        embedding_batch = self._get_embedding_batch(
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
        out = self.sigmoid(logits)
        return out

    def extract_mean_embedding(self, x):
        target_embedding = self.forward_target(x)
        context_embedding = self.forward_context(x)
        return (target_embedding + context_embedding) / 2
