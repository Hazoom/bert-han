import argparse
import json
import logging

import _jsonnet
import pypeln as pl
from pathos.multiprocessing import cpu_count


# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from src.datasets import yahoo_dataset
# noinspection PyUnresolvedReferences
from src.models import han
# noinspection PyUnresolvedReferences
from src.models.preprocessors import han_preprocessor
# noinspection PyUnresolvedReferences
from src.nlp import glove_embeddings, spacynlp
# noinspection PyUnresolvedReferences
from src.utils import registry
# noinspection PyUnresolvedReferences
from src.utils import vocab


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _add_item(index: int, item, section, model_preprocessor, total_length):
    to_add, validation_info = model_preprocessor.validate_item(item, section)
    if to_add:
        model_preprocessor.add_item(item, section, validation_info)

    if index and index % 10000 == 0:
        logger.info(f"Finished pre-processing {index} out of {total_length}")


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preprocessor = registry.instantiate(
            registry.lookup('model', config['model']).Preprocessor,
            config['model'],
            unused_keys=("sentence_attention", "word_attention", "name"),
        )

    def preprocess(self):
        self.model_preprocessor.clear_items()
        for section in self.config['dataset']:
            data = registry.construct('dataset', self.config['dataset'][section])
            data = [item for item in enumerate(data)]
            workers = cpu_count()
            logger.info(f"pr-processing section: {section}")
            (
                pl.process.each(
                    lambda x: _add_item(x[0], x[1], section, self.model_preprocessor, len(data)),
                    data,
                    workers=workers,
                    maxsize=0
                )
                | list
            )
            logger.info(f"Finished pre-processing {len(data)} out of {len(data)}")
        self.model_preprocessor.save()


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config)
    preprocessor.preprocess()


if __name__ == '__main__':
    main(add_parser())
