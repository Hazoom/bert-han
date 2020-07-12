import argparse
import json

import _jsonnet
import tqdm


# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from src.utils import registry
# noinspection PyUnresolvedReferences
from src.utils import vocab
# noinspection PyUnresolvedReferences
from src import models
# noinspection PyUnresolvedReferences
from src import datasets
# noinspection PyUnresolvedReferences
from src import embeddings


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preprocessor = registry.instantiate(
            registry.lookup('model', config['model']).Preprocessor,
            config['model'])

    def preprocess(self):
        self.model_preprocessor.clear_items()
        for section in self.config['data']:
            data = registry.construct('dataset', self.config['data'][section])
            for item in tqdm.tqdm(data, desc=f"{section} section", dynamic_ncols=True):
                to_add, validation_info = self.model_preprocessor.validate_item(item, section)
                if to_add:
                    self.model_preprocessor.add_item(item, section, validation_info)
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
