import torch
import torch.nn
import torch.utils.data
from typing import List, Dict

from src.utils.vocab import Vocab


class HANDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            component: List[Dict],
            label_to_id: Dict,
            vocab: Vocab,
            max_sent_length: int,
            max_doc_length: int,
            num_classes: int,
            dataset_size: int,
    ):
        self.max_doc_length = max_doc_length
        self.max_sent_length = max_sent_length
        self.component = component
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.num_classes = num_classes
        self.dataset_size = dataset_size

        self.labels = torch.tensor([self.label_to_id[str(item["label"])] for item in self.component], dtype=torch.long)

    def _transform_text(self, sentences: List[List[str]]):
        # encode document with max sentence length and max document length (maximum number of sentences)
        doc = [self.vocab.indices(sentence) for sentence in sentences]
        doc = [sent[:self.max_sent_length] for sent in doc][:self.max_doc_length]
        num_sents = min(len(doc), self.max_doc_length)

        # skip erroneous ones
        if not num_sents:
            return None, -1, None

        num_words = [min(len(sent), self.max_sent_length) for sent in doc][:self.max_doc_length]

        return doc, num_sents, num_words

    def __getitem__(self, i):
        text: List[List[str]] = self.component[i]["sentences"]

        doc, num_sents, num_words = self._transform_text(text)

        if num_sents == -1:
            return None

        return doc, self.labels[i].tolist(), num_sents, num_words

    def __len__(self):
        return self.dataset_size


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, doc_lengths, sent_lengths = list(zip(*batch))

    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])

    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()

    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.tensor(sent_lengths[doc_idx], dtype=torch.long)
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length] = torch.tensor(sent, dtype=torch.long)

    return (
        docs_tensor,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(doc_lengths, dtype=torch.long),
        sent_lengths_tensor,
        None,
    )
