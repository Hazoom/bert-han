import os
import json
import collections
import itertools
from typing import Tuple

from src.nlp.abstract_embeddings import Embedder
from src.nlp.abstract_nlp import NLP
from src.models import abstract_preprocessor
from src.utils import registry, vocab
from src.datasets.hanitem import HANItem


@registry.register('preprocessor', 'HANPreprocessor')
class HANPreprocessor(abstract_preprocessor.AbstractPreproc):
    def __init__(self, save_path, min_freq, max_count, word_emb, nlp):
        self.word_emb: Embedder = registry.instantiate(
            registry.lookup("word_emb", word_emb["name"]),
            word_emb,
            unused_keys=("name",),
        )
        self.nlp: NLP = registry.instantiate(
            registry.lookup("nlp", nlp["name"]),
            nlp,
            unused_keys=("name",),
        )

        self.data_dir = os.path.join(save_path, 'han')
        self.texts = collections.defaultdict(list)

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(save_path, 'han_vocab.json')
        self.vocab_word_freq_path = os.path.join(save_path, 'han_word_freq.json')
        self.vocab = None
        self.counted_db_ids = set()
        self.preprocessed_schemas = {}

    def validate_item(self, item: HANItem, section: str) -> Tuple[bool, str]:
        return True, ""

    def add_item(self, item: HANItem, section: str, validation_info: str):
        preprocessed = self.preprocess_item(item)
        self.texts[section].append(preprocessed)

        if section == 'train':
            to_count = itertools.chain(preprocessed["sentences"])

            for token in to_count:
                self.vocab_builder.add_word(token)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item: HANItem):
        sentences = [self._tokenize(sentence) for sentence in item.sentences]

        return {
            "raw_sentences": item.sentences,
            "sentences": [sentence for sentence_list in sentences for sentence in sentence_list],
            "label": item.label
        }

    def _tokenize(self, sentence: str):
        return self.nlp.tokenize(sentence)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in vocab")
        self.vocab.save(self.vocab_path)
        self.vocab_builder.save(self.vocab_word_freq_path)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)
        self.vocab_builder.load(self.vocab_word_freq_path)

    def dataset(self, section: str):
        return [json.loads(line) for line in open(os.path.join(self.data_dir, section + '.jsonl'))]