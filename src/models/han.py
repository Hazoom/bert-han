import torch
import torch.utils.data
from typing import Tuple

from nlp import abstract_embeddings
from src.models import abstract_preprocessor
from src.utils import registry, vocab
from src.datasets.hanitem import HANItem


class HANDataset(torch.utils.data.Dataset):
    def __init__(self, component):
        self.component = component

    def __getitem__(self, idx):
        return self.component[idx]

    def __len__(self):
        return len(self.component)


@registry.register('model', 'HAN')
class HANModel(torch.nn.Module):
    class Preprocessor(abstract_preprocessor.AbstractPreproc):

        def __init__(self, preprocessor):
            super().__init__()

            self.preprocessor: abstract_preprocessor.AbstractPreproc = registry.instantiate(
                registry.lookup('preprocessor', preprocessor['name']),
                preprocessor,
                unused_keys=("name",)
            )

        def vocab(self) -> vocab.Vocab:
            return self.preprocessor.vocab()

        def embedder(self) -> abstract_embeddings.Embedder:
            return self.embedder()

        def validate_item(self, item: HANItem, section: str) -> Tuple[bool, str]:
            item_result, validation_info = self.preprocessor.validate_item(item, section)

            return item_result, validation_info

        def add_item(self, item: HANItem, section: str, validation_info: str):
            self.preprocessor.add_item(item, section, validation_info)

        def clear_items(self) -> None:
            self.preprocessor.clear_items()

        def save(self) -> None:
            self.preprocessor.save()

        def load(self) -> None:
            self.preprocessor.load()

        def dataset(self, section) -> HANDataset:
            return HANDataset(self.preprocessor.dataset(section))

    def __init__(self, preprocessor, device, word_attention, sentence_attention):
        super().__init__()
        self.preprocessor = preprocessor
        self.word_attention = registry.construct(
            'word_attention', word_attention, device=device, preproc=preprocessor.preprocessor)
        self.sentence_attention = registry.construct(
            'sentence_attention', sentence_attention, device=device, preproc=preprocessor.preprocessor)

        self.compute_loss = self._compute_loss_enc_batched

    def _compute_loss_batched(self, batch, debug=False):
        losses = []
        enc_states = self.encoder([enc_input for enc_input, dec_output in batch])

        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def eval_on_batch(self, batch):
        mean_loss = self.compute_loss(batch).item()
        batch_size = len(batch)
        result = {'loss': mean_loss * batch_size, 'total': batch_size}
        return result

    def begin_inference(self, orig_item, preproc_item):
        enc_input, _ = preproc_item
        enc_state, = self.encoder([enc_input])
        return self.decoder.begin_inference(enc_state, orig_item)
