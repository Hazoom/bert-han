local _base = import 'han-ag-news.libsonnet';
local _data_path = 'data/ag_news_csv/';

function(args, data_path=_data_path) _base(output_from=true, data_path=data_path) + {
    local lr = 0.000743552663260837,
    local end_lr = 0,
    local bs = 32,
    local seed = 42,

    local lr_s = '%0.1e' % lr,
    local end_lr_s = '0e0',
    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s' % ({
        bs: bs,
        lr: lr_s,
        end_lr: end_lr_s,
    }),

    model+: {
        word_attention+: {
            dropout: 0.2,
            word_emb_size: 300,
            recurrent_size: 256,
            attention_dim: 256,
        },
        sentence_attention+: {
            dropout: 0.2,
            word_recurrent_size: 256,
            recurrent_size: 256,
            attention_dim: 256,
        },
        preprocessor+: {
            word_emb: {
                name: 'glove',
                kind: '42B',
            },
            nlp: {
                name: 'spaCy',
                model: 'en_core_web_sm',
                lemmatize: false,
            },
            min_freq: 5,
            max_count: 30000,
        },
    },

    train+: {
        batch_size: bs,

        model_seed: seed,
        data_seed:  seed,
        init_seed:  seed,
    },

    lr_scheduler+: {
        start_lr: lr,
        end_lr: end_lr,
    },
}
