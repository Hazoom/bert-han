import functools
import spacy

from src.nlp import abstract_nlp
from src.utils import registry


@registry.register("nlp", "spaCy")
class SpaCyNLP(abstract_nlp.NLP):
    def __init__(self, model="en_core_web_sm", lemmatize=False):
        self.nlp = spacy.load(model)
        self.lemmatize = lemmatize

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        doc = self.nlp(text)
        if self.lemmatize:
            return [[tok.lemma_ for tok in sent] for sent in doc.sents]
        else:
            return [[tok.text.lower() for tok in sent] for sent in doc.sents]
