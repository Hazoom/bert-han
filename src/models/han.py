import torch
from torch import nn
from typing import Tuple

from src.nlp import abstract_embeddings
from src.models import abstract_preprocessor
from src.utils import registry, vocab
from src.datasets.hanitem import HANItem
from src.datasets.han_dataset import HANDataset


@registry.register('model', 'HAN')
class HANModel(torch.nn.Module):
    class Preprocessor(abstract_preprocessor.AbstractPreproc):
        def __init__(self, preprocessor):
            super().__init__()

            self.preprocessor: abstract_preprocessor.AbstractPreproc = registry.instantiate(
                callable=registry.lookup("preprocessor", preprocessor["name"]),
                config=preprocessor,
                unused_keys=("name",)
            )

        def get_vocab(self) -> vocab.Vocab:
            return self.preprocessor.get_vocab()

        def get_embedder(self) -> abstract_embeddings.Embedder:
            return self.get_embedder()

        def get_num_classes(self) -> int:
            return self.preprocessor.get_num_classes()

        def get_max_doc_length(self) -> int:
            return self.preprocessor.get_max_doc_length()

        def get_max_sent_length(self) -> int:
            return self.preprocessor.get_max_sent_length()

        def validate_item(self, item: HANItem, section: str) -> Tuple[bool, str]:
            item_result, validation_info = self.preprocessor.validate_item(item, section)

            return item_result, validation_info

        def add_item(self, item: HANItem, section: str, validation_info: str):
            self.preprocessor.add_item(item, section, validation_info)

        def clear_items(self) -> None:
            self.preprocessor.clear_items()

        def save(self) -> None:
            self.preprocessor.save()

        def load(self) -> None:
            self.preprocessor.load()

        def dataset(self, section) -> HANDataset:
            return HANDataset(
                self.preprocessor.dataset(section),
                self.get_vocab(),
                self.get_max_sent_length(),
                self.get_max_doc_length(),
            )

    def __init__(self, preprocessor, device, word_attention, sentence_attention):
        super().__init__()
        self.preprocessor = preprocessor
        self.word_attention = registry.instantiate(
            callable=registry.lookup("word_attention", word_attention["name"]),
            config=word_attention,
            unused_keys=("name",),
            device=device,
            preprocessor=preprocessor.preprocessor
        )
        self.sentence_attention = registry.instantiate(
            callable=registry.lookup("sentence_attention", sentence_attention["name"]),
            config=sentence_attention,
            unused_keys=("name",),
            device=device,
            preprocessor=preprocessor.preprocessor
        )

        self.fc = nn.Linear(self.sentence_attention.recurrent_size, preprocessor.preprocessor.get_num_classes())

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(device)

    def forward(self, docs, doc_lengths, sent_lengths, labels):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :param labels: labels; LongTensor (num_docs)
        :return: class scores, loss, attention weights of words, attention weights of sentences
        """
        doc_embeds, word_att_weights, sent_att_weights = self.sent_attention(docs, doc_lengths, sent_lengths)

        scores = self.fc(doc_embeds)

        loss = self.loss(scores, labels)

        return scores, loss, word_att_weights, sent_att_weights
