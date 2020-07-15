import argparse
import json
import logging

import _jsonnet
import tqdm


# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from src.datasets import yahoo_dataset, ag_news_dataset
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


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preprocessor = registry.instantiate(
            callable=registry.lookup("model", config["model"]).Preprocessor,
            config=config["model"],
            unused_keys=("sentence_attention", "word_attention", "name", "final_layer_dim", "final_layer_dropout"),
        )

    def preprocess(self):
        self.model_preprocessor.clear_items()
        for section in self.config["dataset"]:
            data = registry.construct("dataset", self.config["dataset"][section])
            for item in tqdm.tqdm(data, desc=f"pre-processing {section} section", dynamic_ncols=True):
                to_add, validation_info = self.model_preprocessor.validate_item(item, section)
                if to_add:
                    self.model_preprocessor.add_item(item, section, validation_info)
        self.model_preprocessor.save()


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config)
    preprocessor.preprocess()


if __name__ == '__main__':
    main(add_parser())
