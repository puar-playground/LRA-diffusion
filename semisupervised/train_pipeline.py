from typing import Callable, Dict

from model_diffusion import Diffusion
from semisupervised.dataset_utils import LabeledDataset, UnlabeledDataset
from utils.data_utils import Custom_dataset


def train(model_instantiator: Callable[[], Diffusion], labeled_dataset: LabeledDataset,
          unlabeled_dataset: UnlabeledDataset, test_dataset: Custom_dataset, args):
    while not has_converged(unlabeled_dataset):
        model = model_instantiator()
        train_iteration(model, labeled_dataset, args)
        test(model, test_dataset)
        annotated_dataset = annotate_data(model, unlabeled_dataset)
        if len(annotated_dataset) == 0:
            raise Exception("cannot converge")
        labeled_dataset.add_pseudo_labels(annotated_dataset)
        unlabeled_dataset.remove_data_points(annotated_dataset)


def train_iteration(model: Diffusion, labeled_dataset: LabeledDataset, args):
    pass

# @save_model
def test(model: Diffusion, test_dataset: Custom_dataset):
    pass


def annotate_data(model: Diffusion, unlabeled_dataset: UnlabeledDataset) -> Dict[int, int]:
    pass

def has_converged(unlabeled_dataset: UnlabeledDataset):
    pass
