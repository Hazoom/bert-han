import os
import json
import collections
from typing import Tuple, Dict
import random
import srsly
from transformers import BertTokenizer

from src.nlp.abstract_embeddings import Embedder
from src.models import abstract_preprocessor
from src.utils import registry, vocab
from src.datasets.hanitem import HANItem
from src.nlp import textcleaning


@registry.register('preprocessor', 'BERTPreprocessor')
class BERTPreprocessor(abstract_preprocessor.AbstractPreproc):

    def __init__(self, save_path, max_sent_length, max_doc_length, bert_version="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

        self.max_doc_length = max_doc_length
        self.max_sent_length = max_sent_length

        self.data_dir = os.path.join(save_path, "tokenized_data")
        self.texts = collections.defaultdict(list)

        self.classes_path = os.path.join(save_path, "classes.json")
        self.dataset_sizes_path = os.path.join(save_path, "dataset_sizes.json")
        self.label_to_id = {}
        self.dataset_sizes = {}

    def get_vocab(self) -> vocab.Vocab:
        return None

    def get_dataset_size(self, section: str) -> int:
        return self.dataset_sizes[section]

    def get_num_classes(self) -> int:
        return len(self.label_to_id)

    def get_embedder(self) -> Embedder:
        return None

    def get_max_sent_length(self) -> int:
        return self.max_sent_length

    def get_max_doc_length(self) -> int:
        return self.max_doc_length

    def validate_item(self, item: HANItem, section: str) -> Tuple[bool, str]:
        return True, ""

    def add_item(self, item: HANItem, section: str, validation_info: str):
        preprocessed = self.preprocess_item(item)
        self.texts[section].append(preprocessed)
        if preprocessed["label"] not in self.label_to_id:
            self.label_to_id[preprocessed["label"]] = int(preprocessed["label"]) - 1

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item: HANItem):
        sentences = [textcleaning.clean_text(sentence) for sentence in item.sentences]
        sentences = [sentence.split("\\n") for sentence in sentences]
        sentences = [sentence for sentence_list in sentences for sentence in sentence_list if sentence]
        sentences = [self._tokenize(sentence) for sentence in sentences if sentence]

        return {
            "sentences": sentences,
            "label": int(item.label),
        }

    def _tokenize(self, sentence: str):
        return self.tokenizer.tokenize(sentence)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.data_dir)

        with open(self.classes_path, "w") as out_fp:
            json.dump(self.label_to_id, out_fp)

        with open(self.dataset_sizes_path, "w") as out_fp:
            json.dump({section: len(texts) for section, texts in self.texts.items()}, out_fp)

        for section, texts in self.texts.items():
            if section == "train":
                # sort documents by the number of sentences for faster training
                texts = sorted(texts, key=lambda x: len(x["sentences"]), reverse=True)
            srsly.write_jsonl(os.path.join(self.data_dir, section + ".jsonl"), texts)

    def load(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.data_dir)

        with open(self.classes_path, "r") as in_fp:
            self.label_to_id = json.load(in_fp)

        with open(self.dataset_sizes_path, "r") as in_fp:
            self.dataset_sizes = json.load(in_fp)

    def label_to_id_map(self) -> Dict:
        return self.label_to_id

    def dataset(self, section: str):
        return list(srsly.read_jsonl(os.path.join(self.data_dir, section + ".jsonl")))

    def create_validation_set(self, val_split: float, path: str) -> None:
        print(f"creating validation set with split: {val_split} to path: {path}")
        texts = self.texts["train"]
        sample_indices = set(random.sample(range(len(texts)), int(val_split * len(texts))))
        val_texts = [item for item_index, item in enumerate(texts) if item_index in sample_indices]
        train_texts = [item for item_index, item in enumerate(texts) if item_index not in sample_indices]
        self.texts["train"] = train_texts
        self.texts["val"] = val_texts

    def get_tokenizer(self):
        return self.tokenizer
