import os
import argparse
import itertools
import json

import tqdm
import _jsonnet
import torch
import torch.utils.data

from sklearn.metrics import classification_report, accuracy_score

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from src.datasets import yahoo_dataset, ag_news_dataset, classes
# noinspection PyUnresolvedReferences
from src.models import han, wordattention, sentenceattention, optimizers, bert_wordattention
# noinspection PyUnresolvedReferences
from src.models.preprocessors import han_preprocessor, bert_preprocessor
# noinspection PyUnresolvedReferences
from src.nlp import glove_embeddings, spacynlp
# noinspection PyUnresolvedReferences
from src.utils import registry
# noinspection PyUnresolvedReferences
from src.utils import vocab

from src.utils import registry
from src.utils import saver as saver_mod
from src.datasets.han_dataset import collate_fn


class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)

        # 0. Construct classes dictionary mapping
        self.classes = registry.instantiate(
            callable=registry.lookup("classes", config["classes"]),
            config=config["classes"],
            unused_keys=("name",),
        )

        self.label_to_name = self.classes.get_classes_dict()

        # 1. Construct preprocessors
        self.model_preprocessor = registry.instantiate(
            callable=registry.lookup("model", config["model"]).Preprocessor,
            config=config["model"],
            unused_keys=(
                "model", "name", "sentence_attention", "word_attention", "final_layer_dim", "final_layer_dropout"
            ),
        )
        self.model_preprocessor.load()

        self.id_to_label = {value: key for key, value in self.model_preprocessor.label_to_id_map().items()}

    def load_model(self, logdir):
        """Load a model (identified by the config used for construction) and return it"""
        # 1. Construct model
        model = registry.construct(
            kind="model",
            config=self.config["model"],
            unused_keys=("preprocessor",),
            preprocessor=self.model_preprocessor,
            device=self.device,
        )
        model.to(self.device)
        model.eval()

        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, map_location=self.device, item_keys=["model"])
        if not last_step:
            raise Exception(f"Attempting to infer on untrained model in {logdir}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, "w")

        with torch.no_grad():
            orig_data = registry.construct("dataset", self.config["dataset"][args.section])
            preproc_data = self.model_preprocessor.dataset(args.section)
            if args.limit:
                sliced_preproc_data = itertools.islice(preproc_data, args.limit)
            else:
                sliced_preproc_data = preproc_data
            assert len(orig_data) == len(preproc_data)
            self._inner_infer(model, preproc_data, sliced_preproc_data, output)

    def _inner_infer(self, model, preproc_data, sliced_preproc_data, output, batch_size=32):
        test_data_loader = torch.utils.data.DataLoader(
            sliced_preproc_data,
            batch_size=batch_size,
            collate_fn=preproc_data.bert_collate_fn if "bert" in str(
                        type(model.preprocessor.preprocessor)).lower() else collate_fn)

        true_labels = []
        predictions_labels = []
        index = 0
        model.eval()
        with torch.no_grad():
            for test_batch in tqdm.tqdm(test_data_loader, total=len(test_data_loader)):
                docs, labels, doc_lengths, sent_lengths, additional_data = test_batch

                docs = docs.to(self.device)  # (batch_size, padded_doc_length, padded_sent_length)
                labels = labels.to(self.device)  # (batch_size)
                sent_lengths = sent_lengths.to(self.device)  # (batch_size, padded_doc_length)
                doc_lengths = doc_lengths.to(self.device)  # (batch_size)

                attention_masks = None
                token_type_ids = None
                if additional_data:
                    attention_masks = additional_data["attention_masks"].to(self.device)
                    token_type_ids = additional_data["token_type_ids"].to(self.device)

                # scores: (n_docs, n_classes)
                # word_att_weights: (n_docs, max_doc_len_in_batch, max_sent_len_in_batch)
                # sentence_att_weights: (n_docs, max_doc_len_in_batch)
                # loss: float
                if attention_masks is not None and token_type_ids is not None:
                    scores, word_att_weights, sentence_att_weights, loss = model(
                        docs, doc_lengths, sent_lengths, labels, attention_masks, token_type_ids
                    )
                else:
                    scores, word_att_weights, sentence_att_weights, loss = model(
                        docs, doc_lengths, sent_lengths, labels
                    )

                predictions = scores.max(dim=1)[1].tolist()
                scores = scores.tolist()
                word_att_weights = word_att_weights.tolist()
                sentence_att_weights = sentence_att_weights.tolist()

                for i in range(index, min(index + batch_size, len(sliced_preproc_data.component))):
                    index_in_batch = i - index
                    true_label_int = int(sliced_preproc_data.component[i]["label"])
                    predicted_label_int = int(self.id_to_label[predictions[index_in_batch]])
                    output.write(
                        json.dumps({
                            "index": index_in_batch,
                            "original_document": sliced_preproc_data.component[i]["sentences"],
                            "true_label": self.label_to_name[true_label_int],
                            "predicted_label": self.label_to_name[predicted_label_int],
                            "probs": scores[index_in_batch],
                            "word_att_weights": word_att_weights[index_in_batch],
                            "sentence_att_weights": sentence_att_weights[index_in_batch],
                        }) + "\n")
                    output.flush()
                    true_labels.append(true_label_int - 1)
                    predictions_labels.append(predicted_label_int - 1)

                index += batch_size

        target_names = [self.label_to_name[key] for key in sorted(self.label_to_name.keys())]
        print("Test accuracy:", accuracy_score(true_labels, predictions_labels))
        report = classification_report(true_labels, predictions_labels, target_names=target_names)
        print("Classification Report:\n", report, "\n")


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')

    parser.add_argument('--section', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if "model_name" in config:
        args.logdir = os.path.join(args.logdir, config["model_name"])

    output_path = args.output.replace("__LOGDIR__", args.logdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir)
    inferer.infer(model, output_path, args)


if __name__ == "__main__":
    main(add_parser())
