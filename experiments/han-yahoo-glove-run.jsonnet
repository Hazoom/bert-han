{
    logdir: "logdir/glove_run",
    model_config: "configs/han-yahoo-glove.jsonnet",
    model_config_args: {
    },

    eval_name: "glove_run",
    eval_output: "__LOGDIR__/ie_dirs",
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}