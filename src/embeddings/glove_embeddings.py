import abc
import functools
import os

import corenlp
import torchtext

from src.utils import registry


class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, sentence):
        """Given a string, return a list of tokens suitable for lookup."""
        pass

    @abc.abstractmethod
    def lookup(self, token):
        """Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None."""
        pass

    @abc.abstractmethod
    def contains(self, token) -> bool:
        pass

    @abc.abstractmethod
    def to(self, device):
        """Transfer the pretrained embeddings to the given device."""
        pass


@registry.register("word_emb", "glove")
class GloVe(Embedder):

    def __init__(self, kind, lemmatize=False):
        cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors
        self.lemmatize = lemmatize
        self.corenlp_annotators = ['tokenize', 'ssplit']
        if lemmatize:
            self.corenlp_annotators.append('lemma')

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        ann = corenlp.annotate(text, self.corenlp_annotators)
        if self.lemmatize:
            return [[tok.lemma.lower() for tok in sent.token] for sent in ann.sentence]
        else:
            return [[tok.word.lower() for tok in sent.token] for sent in ann.sentence]

    def lookup(self, token):
        index = self.glove.stoi.get(token)
        return self.vectors[index] if index is not None else None

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)
