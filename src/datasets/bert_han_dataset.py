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

    def _pad_sequence_for_bert_batch(self, tokens_lists, max_len: int):
        pad_id = self.tokenizer.pad_token_id
        assert max_len <= self.max_sent_length
        toks_ids = []
        att_masks = []
        tok_type_lists = []
        for item_toks in tokens_lists:
            padded_item_toks = item_toks + [pad_id] * (max_len - len(item_toks))
            toks_ids.append(padded_item_toks)

            _att_mask = [1] * len(item_toks) + [0] * (max_len - len(item_toks))
            att_masks.append(_att_mask)

            first_sep_id = padded_item_toks.index(self.tokenizer.sep_token_id)
            assert first_sep_id > 0
            _tok_type_list = [0] * (first_sep_id + 1) + [1] * (max_len - first_sep_id - 1)
            tok_type_lists.append(_tok_type_list)
        return toks_ids, att_masks, tok_type_lists

    def _transform_text(self, sentences: List[List[str]]):
        # encode document with max sentence length and max document length (maximum number of sentences)
        docs = []
        for sentence in sentences:
            bert_sentence = self._pad_single_sentence_for_bert(sentence)
            if len(bert_sentence) <= self.max_sent_length:
                docs.append(bert_sentence)
        docs = docs[:self.max_doc_length]
        num_sents = min(len(docs), self.max_doc_length)

        # skip erroneous ones
        if not num_sents:
            return None, -1, None

        num_words = [min(len(sent), self.max_sent_length) for sent in docs][:self.max_doc_length]

        return docs, num_sents, num_words

    def __getitem__(self, i):
        text: List[List[str]] = self.component[i]["sentences"]

        doc, num_sents, num_words = self._transform_text(text)

        if num_sents == -1:
            return None

        return doc, self.labels[i].tolist(), num_sents, num_words

    def __len__(self):
        return self.dataset_size

    def bert_collate_fn(self, batch):
        batch = filter(lambda x: x is not None, batch)
        docs, labels, doc_lengths, sent_lengths = list(zip(*batch))

        docs = [[self.tokenizer.convert_tokens_to_ids(sentence) for sentence in doc] for doc in docs]

        bsz = len(labels)
        batch_max_doc_length = max(doc_lengths)
        batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])

        docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length), dtype=torch.long)
        batch_att_mask_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length), dtype=torch.long)
        token_type_ids_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length), dtype=torch.long)
        sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length))

        for doc_idx, doc in enumerate(docs):
            padded_token_lists, att_mask_lists, tok_type_lists = self._pad_sequence_for_bert_batch(
                doc, batch_max_sent_length
            )

            doc_length = doc_lengths[doc_idx]
            sent_lengths_tensor[doc_idx, :doc_length] = torch.tensor(sent_lengths[doc_idx], dtype=torch.long)

            for sent_idx, (padded_tokens, att_masks, tok_types) in enumerate(
                    zip(padded_token_lists, att_mask_lists, tok_type_lists)):
                docs_tensor[doc_idx, sent_idx, :] = torch.tensor(padded_tokens, dtype=torch.long)
                batch_att_mask_tensor[doc_idx, sent_idx, :] = torch.tensor(att_masks, dtype=torch.long)
                token_type_ids_tensor[doc_idx, sent_idx, :] = torch.tensor(tok_types, dtype=torch.long)

        return (
            docs_tensor,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(doc_lengths, dtype=torch.long),
            sent_lengths_tensor,
            dict(attention_masks=batch_att_mask_tensor, token_type_ids=token_type_ids_tensor),
        )
