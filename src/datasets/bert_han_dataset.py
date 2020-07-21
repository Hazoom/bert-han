import torch
import torch.nn
import torch.utils.data
from typing import List, Dict

from transformers import BertTokenizer


class BERTHANDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            component: List[Dict],
            label_to_id: Dict,
            max_sent_length: int,
            max_doc_length: int,
            num_classes: int,
            dataset_size: int,
            tokenizer: BertTokenizer,
    ):
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.max_sent_length = max_sent_length
        self.component = component
        self.label_to_id = label_to_id
        self.num_classes = num_classes
        self.dataset_size = dataset_size

        self.labels = torch.tensor([self.label_to_id[str(item["label"])] for item in self.component], dtype=torch.long)

    def _pad_single_sentence_for_bert(self, tokens: List[str], cls: bool = True):
        if cls:
            return [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        else:
            return tokens + [self.tokenizer.sep_token]

    def _pad_sequence_for_bert_batch(self, tokens_lists):
        pad_id = self.tokenizer.pad_token_id
        max_len = max([len(it) for it in tokens_lists])
        assert max_len <= self.max_sent_length
        toks_ids = []
        att_masks = []
        for item_toks in tokens_lists:
            padded_item_toks = item_toks + [pad_id] * (max_len - len(item_toks))
            toks_ids.append(padded_item_toks)

            _att_mask = [1] * len(item_toks) + [0] * (max_len - len(item_toks))
            att_masks.append(_att_mask)
        return toks_ids, att_masks

    def _transform_text(self, sentences: List[List[str]]):
        # encode document with max sentence length and max document length (maximum number of sentences)
        docs = []
        for sentence in sentences:
            bert_sentence = self._pad_single_sentence_for_bert(sentence)
            if len(bert_sentence) <= self.max_sent_length:
                docs.append(bert_sentence)
        docs = [self._pad_single_sentence_for_bert(sentence) for sentence in sentences]
        docs = [sent[:self.max_sent_length] for sent in docs][:self.max_doc_length]
        num_sents = min(len(docs), self.max_doc_length)

        # skip erroneous ones
        if not num_sents:
            return None, -1, None

        num_words = [min(len(sent), self.max_sent_length) for sent in docs][:self.max_doc_length]

        padded_token_lists, _ = self._pad_sequence_for_bert_batch(docs)

        return padded_token_lists, num_sents, num_words

    def __getitem__(self, i):
        text: List[List[str]] = self.component[i]["sentences"]

        doc, num_sents, num_words = self._transform_text(text)

        if num_sents == -1:
            return None

        return doc, self.labels[i].tolist(), num_sents, num_words

    def __len__(self):
        return self.dataset_size
