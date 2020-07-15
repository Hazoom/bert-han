import csv
import tqdm
import torch.utils.data

from src.datasets.hanitem import HANItem
from src.utils import registry


@registry.register("dataset", "ag_news")
class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, path, limit=None):
        self.path = path

        self.items = []
        with open(path, "r") as in_file:
            reader = csv.reader(in_file)
            for index, line in enumerate(tqdm.tqdm(reader, desc=f"reading rows from path: {path}", dynamic_ncols=True)):
                if index > 0:
                    self.items.append(HANItem(sentences=[line[1], line[2]], label=line[0]))
                if limit and len(self.items) >= limit:
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
