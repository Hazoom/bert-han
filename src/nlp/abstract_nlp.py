import abc


class NLP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, text: str):
        """
        Tokenize and parse text.
        :param text: str
        """
        pass
