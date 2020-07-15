import abc
from typing import List, Dict


from src.utils import vocab
from src.nlp import abstract_embeddings


class AbstractPreproc(metaclass=abc.ABCMeta):
    """Used for preprocessing data according to the model's liking.

    Some tasks normally performed here:
    - Constructing a vocabulary from the training data
    - Transforming the items in some way
    - Loading and providing the pre-processed data to the model
    """
    
    @abc.abstractmethod
    def validate_item(self, item, section):
        """Checks whether item can be successfully preprocessed.

        Returns a boolean and an arbitrary object."""
        pass

    @abc.abstractmethod
    def add_item(self, item, section, validation_info):
        """Add an item to be preprocessed."""
        pass

    @abc.abstractmethod
    def clear_items(self):
        """Clear the preprocessed items"""
        pass

    @abc.abstractmethod
    def save(self):
        """Marks that all of the items have been preprocessed. Save state to disk.

        Used in preprocess.py, after reading all of the data."""
        pass

    @abc.abstractmethod
    def load(self):
        """Load state from disk."""
        pass

    @abc.abstractmethod
    def dataset(self, section) -> List[Dict]:
        """Returns a torch.data.utils.Dataset instance."""
        pass

    @abc.abstractmethod
    def get_vocab(self) -> vocab.Vocab:
        pass

    @abc.abstractmethod
    def get_embedder(self) -> abstract_embeddings.Embedder:
        pass

    @abc.abstractmethod
    def get_max_doc_length(self) -> int:
        pass

    @abc.abstractmethod
    def get_max_sent_length(self) -> int:
        pass

    @abc.abstractmethod
    def get_num_classes(self) -> int:
        pass

    @abc.abstractmethod
    def label_to_id_map(self) -> Dict:
        pass

    @abc.abstractmethod
    def get_dataset_size(self, section: str) -> int:
        pass
