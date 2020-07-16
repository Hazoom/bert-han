function(output_from, data_path='data/ag_news_csv/') {
    local PREFIX = data_path,
    
    dataset: {
        train: {
            name: 'ag_news',
            path: PREFIX + 'train.csv',
        },
        val: {
            name: 'ag_news',
            path: PREFIX + 'val.csv',
            val_split: 0.1,
        },
        test: {
            name: 'ag_news',
            path: PREFIX + 'test.csv',
        },
    },

    model: {
        name: 'HAN',
        word_attention: {
            name: 'WordAttention',
            dropout: 0.2,
            word_emb_size: 300,
            recurrent_size: 256,
            attention_dim: 256,
        },   
        sentence_attention: {
            name: 'SentenceAttention',
            dropout: 0.2,
            word_recurrent_size: 256,
            recurrent_size: 256,
            attention_dim: 256,
        },
        preprocessor: {
            name: 'HANPreprocessor',
            word_emb: {
                name: 'glove',
                kind: '42B',
            },
            nlp: {
                name: 'spaCy',
                model: 'en_core_web_sm'
            },
            min_freq: 5,
            max_count: 30000,

            save_path: PREFIX + 'han,output_from=%s,emb=glove-42B/' % [output_from],
            max_sent_length: 64,
            max_doc_length: 20,
        },
        final_layer_dim: 50,
        final_layer_dropout: 0.3,
    },

    train: {
        batch_size: 32,
        eval_batch_size: 50,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,

        max_steps: 20000,
        num_train_eval_items: 10000,
    },

    optimizer: {
        name: 'adam',
    },

    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    },

    log: {
        reopen_to_flush: true,
    }
}
