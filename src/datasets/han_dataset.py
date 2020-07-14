from typing import List, Dict
import torch.utils.data

from src.utils.vocab import Vocab


class HANDataset(torch.utils.data.Dataset):
    def __init__(self, component: List[Dict], vocab: Vocab, max_sent_length: int, max_doc_length: int):
        self.max_doc_length = max_doc_length
        self.max_sent_length = max_sent_length
        self.component = component
        self.vocab = vocab

    def transform(self, sentences: List[List[str]]):
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
        label: str = self.component[i]["label"]
        text: List[List[str]] = self.component[i]["sentences"]

        doc, num_sents, num_words = self.transform(text)

        if num_sents == -1:
            return None

        return doc, label, num_sents, num_words

    def __len__(self):
        return len(self.component)
