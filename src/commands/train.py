import argparse
import collections
import datetime
import json
import os

import _jsonnet
import attr
import torch
import torch.utils.data

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from src.datasets import yahoo_dataset
# noinspection PyUnresolvedReferences
from src.models import han, wordattention, sentenceattention, optimizers
# noinspection PyUnresolvedReferences
from src.models.preprocessors import han_preprocessor
# noinspection PyUnresolvedReferences
from src.nlp import glove_embeddings, spacynlp
# noinspection PyUnresolvedReferences
from src.utils import registry
# noinspection PyUnresolvedReferences
from src.utils import vocab

from src.utils import registry, random_state
from src.utils import saver as saver_mod
from src.datasets.han_dataset import collate_fn


def _yield_batches_from_epochs(loader):
    while True:
        for batch in loader:
            yield batch


@attr.s
class TrainConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)

    batch_size = attr.ib(default=32)
    eval_batch_size = attr.ib(default=32)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)

    num_batch_accumulated = attr.ib(default=1)
    clip_grad = attr.ib(default=None)


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, "a+")

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + "\n")
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, "a+")
            else:
                self.log_file.flush()


class Trainer:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.logger = logger
        self.train_config = registry.instantiate(TrainConfig, config["train"])
        self.data_random = random_state.RandomContext(self.train_config.data_seed)
        self.model_random = random_state.RandomContext(self.train_config.model_seed)
        self.init_random = random_state.RandomContext(self.train_config.init_seed)

        with self.init_random:
            # Load preprocessors
            self.model_preprocessor = registry.instantiate(
                callable=registry.lookup("model", config["model"]).Preprocessor,
                config=config["model"],
                unused_keys=(
                    "model", "name", "sentence_attention", "word_attention", "final_layer_dim", "final_layer_dropout"
                ),
            )
            self.model_preprocessor.load()

            # Construct model
            self.model = registry.construct(
                kind="model",
                config=config["model"],
                unused_keys=("preprocessor",),
                preprocessor=self.model_preprocessor,
                device=self.device,
            )
            self.model.to(self.device)

    def train(self, config, model_dir):
        with self.init_random:
            optimizer = registry.construct(
                kind="optimizer",
                config=config["optimizer"],
                params=self.model.parameters(),
            )
            lr_scheduler = registry.construct(
                kind="lr_scheduler",
                config=config["lr_scheduler"],
                param_groups=optimizer.param_groups,
            )

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            {"model": self.model, "optimizer": optimizer},
            keep_every_n=self.train_config.keep_every_n,
        )
        last_step = saver.restore(model_dir, map_location=self.device)

        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preprocessor.dataset("train")
            train_data_loader = _yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=False,
                    drop_last=True,
                    collate_fn=collate_fn))
        train_eval_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.train_config.eval_batch_size,
            collate_fn=collate_fn)

        val_data = self.model_preprocessor.dataset("test")  #TODO: change to validation set
        val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.train_config.eval_batch_size,
            collate_fn=collate_fn)

        # 4. Start training loop
        with self.data_random:
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.train_config.max_steps:
                    break

                # Evaluate model
                if last_step > 0 and last_step % self.train_config.eval_every_n == 0:
                    if self.train_config.eval_on_train:
                        self._eval_model(
                            self.logger,
                            self.model,
                            last_step,
                            train_eval_data_loader,
                            "train",
                            num_eval_items=self.train_config.num_eval_items,
                        )
                    if self.train_config.eval_on_val:
                        self._eval_model(
                            self.logger,
                            self.model,
                            last_step,
                            val_data_loader,
                            "val",
                            num_eval_items=self.train_config.num_eval_items,
                        )

                # Compute and apply gradient
                with self.model_random:
                    docs, labels, doc_lengths, sent_lengths = batch

                    docs = docs.to(self.device)  # (batch_size, padded_doc_length, padded_sent_length)
                    labels = labels.to(self.device)  # (batch_size)
                    sent_lengths = sent_lengths.to(self.device)  # (batch_size, padded_doc_length)
                    doc_lengths = doc_lengths.to(self.device)  # (batch_size)

                    # scores: (n_docs, n_classes)
                    # word_att_weights: (n_docs, max_doc_len_in_batch, max_sent_len_in_batch)
                    # sentence_att_weights: (n_docs, max_doc_len_in_batch)
                    # loss: float
                    scores, word_att_weights, sentence_att_weights, loss = self.model(
                        docs, doc_lengths, sent_lengths, labels
                    )

                    loss.backward()

                    if self.train_config.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            optimizer.bert_param_group["params"], self.train_config.clip_grad
                        )
                    optimizer.step()
                    lr_scheduler.update_lr(last_step)
                    optimizer.zero_grad()

                # Report metrics
                if last_step > 0 and last_step % self.train_config.report_every_n == 0:

                    # Compute accuracy
                    predictions = scores.max(dim=1)[1]
                    acc = torch.eq(predictions, labels).mean().item()

                    self.logger.log(f"Step {last_step}: loss={loss.item():.4f} acc={acc:.4f}")

                last_step += 1

                # Run saver
                if last_step == 1 or last_step % self.train_config.save_every_n == 0:
                    saver.save(model_dir, last_step)

            # Save final model
            saver.save(model_dir, last_step)

    def _eval_model(self, logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
            for eval_batch in eval_data_loader:
                docs, labels, doc_lengths, sent_lengths = eval_batch
                batch_size = len(labels)

                docs = docs.to(self.device)  # (batch_size, padded_doc_length, padded_sent_length)
                labels = labels.to(self.device)  # (batch_size)
                sent_lengths = sent_lengths.to(self.device)  # (batch_size, padded_doc_length)
                doc_lengths = doc_lengths.to(self.device)  # (batch_size)

                # scores: (n_docs, n_classes)
                # loss: float
                # word_att_weights: (n_docs, max_doc_len_in_batch, max_sent_len_in_batch)
                # sentence_att_weights: (n_docs, max_doc_len_in_batch)
                scores, _, _, loss = self.model(
                    docs, doc_lengths, sent_lengths, labels
                )

                predictions = scores.max(dim=1)[1]
                acc = torch.eq(predictions, labels).mean().item()

                mean_loss = loss * batch_size
                mean_acc = acc * batch_size

                stats["loss"] += mean_loss
                stats["acc"] += mean_acc
                stats["total"] += batch_size

                if num_eval_items and stats["total"] > num_eval_items:
                    break
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != "total":
                stats[k] /= stats["total"]
        if "total" in stats:
            del stats["total"]

        kv_stats = ", ".join(f"{k} = {v}" for k, v in stats.items())
        logger.log(f"Step {last_step} stats, {eval_section}: {kv_stats}")


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if "model_name" in config:
        log_dir = os.path.join(args.logdir, config["model_name"])
    else:
        log_dir = args.logir

    # Initialize the logger
    reopen_to_flush = config.get("log", {}).get("reopen_to_flush")
    logger = Logger(os.path.join(log_dir, "log.txt"), reopen_to_flush)

    # Save the config info
    with open(
            os.path.join(log_dir, f"config-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S%Z')}.json"), "w"
    ) as out_fp:
        json.dump(config, out_fp, sort_keys=True, indent=4)

    logger.log(f"Logging to {log_dir}")

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, model_dir=log_dir)


if __name__ == "__main__":
    main(add_parser())
