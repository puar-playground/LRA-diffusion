from utils.data_utils import Custom_dataset
from typing import Dict

class LabeledDataset(Custom_dataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    def add_pseudo_labels(self, pseudo_labels: Dict[int, int]):
        pass


class UnlabeledDataset(Custom_dataset):
    def __init__(self, data):
        super().__init__(data, [None for i in range(len(data))])

    def remove_data_points(self, pseudo_labels: Dict[int, int]):
        pass


def split_dataset(dataset: Custom_dataset, labeled_data_percentage: float) -> (LabeledDataset, UnlabeledDataset):
    pass
