import logging
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import optim, nn
from torch.utils import data
from tqdm import tqdm

from model_diffusion import Diffusion
from semisupervised.dataset_utils import LabeledDataset, UnlabeledDataset
from semisupervised.logging_utils import LogIndividualCallToFile
from utils.data_utils import Custom_dataset
from utils.ema import EMA
from utils.knn_utils import sample_knn_labels
from utils.learning import prepare_fp_x, cast_label_to_one_hot_and_prototype, adjust_learning_rate, predict_labels, \
    count_correct_predictions


@LogIndividualCallToFile
def train(model_instantiator: Callable[[], Diffusion], labeled_dataset: LabeledDataset,
          unlabeled_dataset: UnlabeledDataset, test_dataset: Custom_dataset, args):
    logger = logging.getLogger("train")
    logger.info("Starting training...")
    iteration_counter = 0
    while not has_converged(unlabeled_dataset):
        if iteration_counter >= 3:
            print("reached max iterations")
            exit()
        model = model_instantiator()
        iteration_counter += 1
        logger.info(f"Training iteration {iteration_counter} with {len(labeled_dataset)} labeled samples")
        train_iteration(model, labeled_dataset, args)
        metrics_evaluation = test(model, test_dataset, args)
        logger.info(f"Evaluation metrics for iteration {iteration_counter}: {metrics_evaluation}")
        # TODO - save model checkpoint

        annotated_dataset = annotate_data(model, unlabeled_dataset, args)
        if len(annotated_dataset) == 0:  # TODO - add hyperparameter for annotation threshold
            logger.error("No unlabeled data has been annotated!")
            raise Exception("cannot converge")
        logger.info(f"Generated {len(annotated_dataset)} confident pseudo-labels from {len(unlabeled_dataset)}"
                    f" unlabeled data samples")
        labeled_dataset.add_pseudo_labels(list(annotated_dataset.values()))
        unlabeled_dataset.remove_data_points(list(annotated_dataset.keys()))
        logger.info(f"{len(unlabeled_dataset)} remaining unlabeled data samples")
    logger.info(f"Finished training with {iteration_counter} iterations")


@LogIndividualCallToFile
def train_iteration(model: Diffusion, labeled_dataset: LabeledDataset, args):
    logger = logging.getLogger("train_iteration")

    device = model.device
    n_class = model.n_class
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs

    train_embed = prepare_fp_x(model.fp_encoder, labeled_dataset, save_dir=None, device=device,
                               fp_dim=2048).to(device)  # TODO fp_dim should not be hardcoded
    train_loader = data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False,
                           eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')
    ema_helper = EMA(mu=0.9999)
    ema_helper.register(model.model)
    model.model.train()

    logger.info(f'Diffusion training start with {len(labeled_dataset)} data samples split into {len(train_loader)} '
                f'batches for {n_epochs} epochs.')
    for epoch in range(n_epochs):
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}',
                  ncols=120, file=open(os.devnull, 'w')) as pbar:
            for i, data_batch in pbar:
                [x_batch, y_batch, _] = data_batch[:3]

                fp_embd = model.fp_encoder(x_batch.to(device))

                # sample a knn labels and compute weight for the sample
                y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch.to(device), train_embed,
                                                                  torch.tensor(labeled_dataset.targets).to(device),
                                                                  k=k, n_class=n_class, weighted=True)

                # convert label to one-hot vector
                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch.to(device)

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs,
                                     n_epochs=1000,
                                     lr_input=0.001)
                n = x_batch.size(0)

                # sampling t
                t = torch.randint(low=0, high=model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = model.forward_t(y_0_batch, x_batch, t, fp_embd)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = torch.matmul(sample_weight, mse_loss)
                loss = torch.mean(weighted_mse_loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(model.model)
                pbar.set_postfix({'loss': loss.item()})
            logger.info(str(pbar))
    print("Finished training.")


# @save_model
def test(model: Diffusion, test_dataset: Custom_dataset, args) -> Dict[str, float]:
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        model.model.eval()
        model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for test_batch_idx, data_batch in enumerate(test_loader):
            [images, target, _] = data_batch[:3]
            target = target.to(args.device)

            predicted_labels, softmax_output = predict_labels(model, images)
            correct = count_correct_predictions(predicted_labels.cpu(), target.cpu())
            correct_cnt += correct
            all_cnt += images.shape[0]

    acc = 100 * correct_cnt / all_cnt
    return {"accuracy": acc}


@LogIndividualCallToFile
def annotate_data(model: Diffusion, unlabeled_dataset: UnlabeledDataset, args) -> Dict[int, Tuple[int, int]]:
    logger = logging.getLogger("annotate_data")

    loader = data.DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    confident_pseudo_labels = dict()
    logger.info(f"Started pseudo-annotating data on {len(unlabeled_dataset)} unlabeled data samples")
    all_probabilities = []
    with torch.no_grad():
        model.model.eval()
        model.fp_encoder.eval()
        for batch_idx, data_batch in enumerate(loader):
            [images, _, indexes] = data_batch[:3]
            predicted_labels, softmax_output = predict_labels(model, images)
            probabilities = get_probabilities_of_pseudo_labels(predicted_labels, softmax_output)
            all_probabilities += probabilities
            confidence = [is_confident_in_label(probability, args) for probability in probabilities]
            confident_labels_in_batch = {key.item(): (unlabeled_dataset.data[key.item()], label.item()) for
                                         key, label, conf_status in
                                         zip(indexes, predicted_labels, confidence) if conf_status}
            confident_pseudo_labels.update(confident_labels_in_batch)
    logger.debug(f'Computed probabilities {all_probabilities}')
    all_probabilities = np.array(all_probabilities)
    logger.info(
        f"Produced probabilities have mean {np.mean(all_probabilities)} and standard deviation {np.std(all_probabilities)}")
    logger.info(
        f'Generated {len(confident_pseudo_labels)} pseudo-labels from {len(unlabeled_dataset)} unlabeled data samples')
    return confident_pseudo_labels


def has_converged(unlabeled_dataset: UnlabeledDataset):
    return len(unlabeled_dataset) == 0


def get_probabilities_of_pseudo_labels(predicted_labels: torch.Tensor, softmax_output: torch.Tensor) -> List[float]:
    return [softmax_output[i][label.item()].item() for i, label in enumerate(predicted_labels)]


def is_confident_in_label(probability: float, args):
    return probability > args.conf_threshold
