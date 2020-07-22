{
    logdir: "logdir/ag_news_bert_run",
    model_config: "configs/han-ag-news-bert.jsonnet",
    model_config_args: {
        bs: 4,
        bert_version: "bert-large-uncased-whole-word-masking",
        max_steps: 12000,
        lr: 7.44e-4,
        bert_lr: 3e-6,
        end_lr: 0,
    },

    test_name: "bert_run",
    test_output: "__LOGDIR__/ie_dirs",
    test_section: "test",
}