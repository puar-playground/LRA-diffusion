import numpy as np

from utils.data_utils import Custom_dataset
from typing import Dict, List, Tuple


class LabeledDataset(Custom_dataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    def add_pseudo_labels(self, new_data: List[Tuple[int, int]]):
        x = [data[0] for data in new_data]
        y = [data[1] for data in new_data]
        self.data = np.append(self.data, np.array(x), axis=0)
        self.targets = np.append(self.targets, y)
        self.n = len(self.data)
        self.index = list(range(self.n))



class UnlabeledDataset(Custom_dataset):
    def __init__(self, data):
        super().__init__(data, [-1 for _ in range(len(data))])

    def remove_data_points(self, label_indexes: List[int]):
        self.data = [data for i, data in enumerate(self.data) if i not in label_indexes]
        self.targets = [target for i, target in enumerate(self.targets) if i not in label_indexes]
        self.n = len(self.data)
        self.index = list(range(self.n))



def split_dataset(dataset: Custom_dataset, labeled_data_percentage: float) -> (LabeledDataset, UnlabeledDataset):
    pass
