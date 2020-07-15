import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from src.models.abstract_preprocessor import AbstractPreproc

from src.utils import vocab
from src.nlp import abstract_embeddings
from src.utils import registry


@registry.register("word_attention", "WordAttention")
class WordAttention(torch.nn.Module):
    def __init__(
            self,
            device: str,
            preprocessor: AbstractPreproc,
            word_emb_size: int,
            dropout: float,
            recurrent_size: int,
            attention_dim: int,
    ):
        super().__init__()
        self._device = device
        self.preprocessor = preprocessor
        self.embedder: abstract_embeddings.Embedder = self.preprocessor.get_embedder()
        self.vocab: vocab.Vocab = self.preprocessor.get_vocab()
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        self.dropout = dropout
        self.attention_dim = attention_dim

        assert self.recurrent_size % 2 == 0
        assert self.word_emb_size == self.embedder.dim

        # embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.word_emb_size)

        # init embedding
        init_embed_list = []
        for index, word in enumerate(self.vocab):
            if self.embedder.contains(word):
                init_embed_list.append(self.embedder.lookup(word))
            else:
                init_embed_list.append(self.embedding.weight[index])
        init_embed_weight = torch.stack(init_embed_list, 0)
        self.embedding.weight = nn.Parameter(init_embed_weight)

        self.encoder = nn.LSTM(
            input_size=self.word_emb_size,
            hidden_size=self.recurrent_size // 2,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Maps LSTM output to `attention_dim` sized tensor
        self.word_weight = nn.Linear(self.recurrent_size, self.attention_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_weight = nn.Linear(self.attention_dim, 1)

    def recurrent_size(self):
        return self.recurrent_size

    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :return: sentences embeddings, docs permutation indices, docs batch sizes, word attention weights
        """

        # Sort documents by decreasing order in length
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim=0, descending=True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `docs` was of size (num_docs, padded_doc_length, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        docs_valid_bsz = packed_sents.batch_sizes

        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_docs, padded_doc_length)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=doc_lengths.tolist(), batch_first=True)

        sents, sent_lengths = packed_sents.data, packed_sent_lengths.data

        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        inp = self.dropout(self.embedding(sents))

        packed_words = pack_padded_sequence(inp, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        sentences_valid_bsz = packed_words.batch_sizes

        # Apply word-level LSTM over word embeddings
        packed_words, _ = self.encoder(packed_words)

        u_i = torch.tanh(self.word_weight(packed_words.data))
        u_w = self.context_weight(u_i).squeeze(1)
        val = u_w.max()
        att = torch.exp(u_w - val)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, sentences_valid_bsz), batch_first=True)

        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        att_weights = att_weights[sent_unperm_idx]

        return sents, doc_perm_idx, docs_valid_bsz, att_weights
