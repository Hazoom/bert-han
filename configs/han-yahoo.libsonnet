function(output_from, data_path='data/yahoo_answers_csv/') {
    local PREFIX = data_path,
    
    dataset: {
        train: {
            name: 'yahoo_answers',
            path: PREFIX + 'train.csv',
        },
        test: {
            name: 'yahoo_answers',
            path: PREFIX + 'test.csv',
        },
    },

    model: {
        name: 'HAN',
        word_attention: {
            name: 'WordAttention',
            dropout: 0.2,
            word_emb_size: 300,
        },   
        sentence_attention: {
            name: 'SentenceAttention',
            dropout: 0.2,
            word_emb_size: 300,
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
        batch_size: 10,
        eval_batch_size: 50,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: 40000,
        num_eval_items: 50,
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
