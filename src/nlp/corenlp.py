import os
import sys
import functools

import corenlp
import requests

from src.nlp import abstract_nlp
from src.utils import registry


@registry.register("nlp", "CoreNLP")
class CoreNLP(abstract_nlp.NLP):
    def __init__(self, model="stanford-corenlp-full-2018-10-05", lemmatize=False):
        if not os.environ.get('CORENLP_HOME'):
            os.environ['CORENLP_HOME'] = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    f'../../third_party/{model}'))
        if not os.path.exists(os.environ['CORENLP_HOME']):
            raise Exception(
                f'''Please install Stanford CoreNLP and put it at {os.environ['CORENLP_HOME']}.

                Direct URL: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
                command: `curl https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip | jar xv`
                Landing page: https://stanfordnlp.github.io/CoreNLP/''')
        self.client = corenlp.CoreNLPClient()
        self.corenlp_annotators = ['tokenize', 'ssplit']
        if lemmatize:
            self.corenlp_annotators.append('lemma')

    def __del__(self):
        if self.client:
            self.client.stop()

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text: str):
        ann = self._annotate(text, self.corenlp_annotators)
        if self.lemmatize:
            return [[tok.lemma.lower() for tok in sent.token] for sent in ann.sentence]
        else:
            return [[tok.word.lower() for tok in sent.token] for sent in ann.sentence]

    def _annotate(self, text, annotators=None, output_format=None, properties=None):
        try:
            result = self.client.annotate(text, annotators, output_format, properties)
        except (corenlp.client.PermanentlyFailedException,
                requests.exceptions.ConnectionError) as e:
            print('\nWARNING: CoreNLP connection timeout. Recreating the server...', file=sys.stderr)
            self.client.stop()
            self.client.start()
            result = self.client.annotate(text, annotators, output_format, properties)

        return result
