import os

import torchtext

from src.nlp import abstract_embeddings
from src.utils import registry


@registry.register("word_emb", "glove")
class GloVe(abstract_embeddings.Embedder):
    def __init__(self, kind):
        cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors

    def dim(self) -> int:
        return self.dim()

    def lookup(self, token):
        index = self.glove.stoi.get(token)
        return self.vectors[index] if index is not None else None

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)
