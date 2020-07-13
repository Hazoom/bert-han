from torch import nn
import torch

from src.models.abstract_preprocessor import AbstractPreproc

from src.utils import registry


@registry.register("word_attention", "WordAttention")
class WordAttention(torch.nn.Module):
    def __init__(
            self, device: str, preprocessor: AbstractPreproc, word_emb_size, dropout, recurrent_size
    ):
        super().__init__()
        self._device = device
        self.preprocessor = preprocessor
        self.embedder = self.preprocessor.embedder()
        self.vocab = preprocessor.vocab()
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        self.dropout = dropout

        assert self.recurrent_size % 2 == 0
        assert self.word_emb_size == self.preprocessor.embedder().dim

        # embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.word_emb_size)

        # init embedding
        init_embed_list = []
        for index, word in enumerate(self.vocab):
            if self.embedder.contains(word):
                init_embed_list.append(self.embedder.lookup(word))
            else:
                init_embed_list.append(self.embedding.weight[index])
        init_embed_weight = torch.stack(init_embed_list, 0)
        self.embedding.weight = nn.Parameter(init_embed_weight)
