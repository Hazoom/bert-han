#!/usr/bin/env python

import argparse
import json

import _jsonnet
import attr


from src.commands import preprocess, train, infer


@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    output = attr.ib()
    limit = attr.ib(default=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval")
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    log_dir = args.logdir or exp_config["logdir"]

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file, model_config_args, log_dir)
        train.main(train_config)
    elif args.mode == "infer":
        infer_output_path = f"{exp_config['test_output']}/{exp_config['test_name']}.jsonl"
        infer_config = InferConfig(
            model_config_file,
            model_config_args,
            log_dir,
            exp_config["test_section"],
            infer_output_path,
        )
        infer.main(infer_config)
    else:
        raise ValueError(f"command {args.mode} is not supported")


if __name__ == "__main__":
    main()
