import abc


class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def lookup(self, token):
        """Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None."""
        pass

    @abc.abstractmethod
    def contains(self, token) -> bool:
        pass

    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @abc.abstractmethod
    def to(self, device):
        """Transfer the pretrained embeddings to the given device."""
        pass
