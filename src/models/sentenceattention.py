from torch import nn
import torch

from src.models.abstract_preprocessor import AbstractPreproc

from src.utils import vocab
from src.nlp import abstract_embeddings
from src.utils import registry


@registry.register("sentence_attention", "SentenceAttention")
class SentenceAttention(torch.nn.Module):
    def __init__(
            self, device: str, preprocessor: AbstractPreproc, word_emb_size: int, dropout: float, recurrent_size: int
    ):
        super().__init__()
        self._device = device
        self.preprocessor = preprocessor
        self.embedder: abstract_embeddings.Embedder = self.preprocessor.get_embedder()
        self.vocab: vocab.Vocab = self.preprocessor.get_vocab()
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        self.dropout = dropout

        assert self.recurrent_size % 2 == 0
        assert self.word_emb_size == self.embedder.dim

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

        self.encoder = nn.LSTM(
            input_size=self.word_emb_size,
            hidden_size=self.recurrent_size // 2,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.word_weight = nn.Parameter(torch.Tensor(self.recurrent_size, self.recurrent_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, self.recurrent_size))
        self.context_weight = nn.Parameter(torch.Tensor(self.recurrent_size, 1))

    def recurrent_size(self):
        return self.recurrent_size

    def forward(self, x):
        inp = self.dropout(self.embedding(x))
        output, (h, c) = self.encoder(inp)
        return output
