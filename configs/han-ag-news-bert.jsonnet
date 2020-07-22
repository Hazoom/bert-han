local _base = import 'han-ag-news.libsonnet';
local _data_path = 'data/ag_news_csv/';

function(args, data_path=_data_path) _base(output_from=true, data_path=data_path) + {
    local data_path = args.data_path,
    local seed = 42,

    local lr_s = '%0.1e' % args.lr,
    local bert_lr_s = '%0.1e' % args.bert_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    local base_bert_enc_size = if args.bert_version == "bert-large-uncased-whole-word-masking" then 1024 else 768,
    local enc_size =  base_bert_enc_size,

    model_name: 'bs=%(bs)d,lr=%(lr)s,bert_lr=%(bert_lr)s,end_lr=%(end_lr)s' % (args + {
        lr: lr_s,
        bert_lr: bert_lr_s,
        end_lr: end_lr_s,
    }),

    model+: {
        word_attention+: {
            name: 'BERTWordAttention',
            bert_version: args.bert_version,
            recurrent_size: base_bert_enc_size,
            attention_dim: base_bert_enc_size,
            word_emb_size: null,
            dropout: null,
        },
        sentence_attention+: {
            recurrent_size: base_bert_enc_size,
            attention_dim: base_bert_enc_size,
            word_recurrent_size: base_bert_enc_size,
        },
        preprocessor+: {
            name: 'BERTPreprocessor',
            save_path: _data_path + 'han,output_from=true,emb=BERT/',
            max_sent_length: 64,
            max_doc_length: 10,
            word_emb: null,
            nlp: null,
            min_freq: null,
            max_count: null,
        },
    },

    train+: {
        batch_size: args.bs,

        model_seed: seed,
        data_seed:  seed,
        init_seed:  seed,
    },

    optimizer: {
        name: 'bertAdamw',
        lr: 0.0,
        bert_lr: 0.0,
    },

    lr_scheduler+: {
        name: 'bert_warmup_polynomial_group',
        start_lrs: [args.lr, args.bert_lr],
        end_lr: args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },
}
