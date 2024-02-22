from typing import Callable, Dict

import torch
from torch import optim, nn
from torch.utils import data

from model_diffusion import Diffusion
from semisupervised.dataset_utils import LabeledDataset, UnlabeledDataset
from utils.data_utils import Custom_dataset
from utils.ema import EMA
from utils.knn_utils import sample_knn_labels
from utils.learning import prepare_fp_x, cast_label_to_one_hot_and_prototype, adjust_learning_rate, cnt_agree


def train(model_instantiator: Callable[[], Diffusion], labeled_dataset: LabeledDataset,
          unlabeled_dataset: UnlabeledDataset, test_dataset: Custom_dataset, args):
    while not has_converged(unlabeled_dataset):
        model = model_instantiator()
        train_iteration(model, labeled_dataset, args)
        test(model, test_dataset, args)
        annotated_dataset = annotate_data(model, unlabeled_dataset)
        if len(annotated_dataset) == 0:
            raise Exception("cannot converge")
        labeled_dataset.add_pseudo_labels(annotated_dataset)
        unlabeled_dataset.remove_data_points(annotated_dataset)


def train_iteration(model: Diffusion, labeled_dataset: LabeledDataset, args):
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

    print(f'Diffusion training start with {len(train_loader)} data samples')
    for epoch in range(n_epochs):
        for i, data_batch in enumerate(train_loader):
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
            adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000,
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
    print("Finished training.")


# @save_model
def test(model: Diffusion, test_dataset: Custom_dataset, args):
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        model.model.eval()
        model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for test_batch_idx, data_batch in enumerate(test_loader):
            [images, target, _] = data_batch[:3]
            target = target.to(args.device)

            label_t_0 = model.reverse_ddim(images, stochastic=False, fq_x=None).detach().cpu()
            correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())
            correct_cnt += correct
            all_cnt += images.shape[0]

    acc = 100 * correct_cnt / all_cnt
    print("Accuracy on test set: ", acc)
    # TODO - save model


def annotate_data(model: Diffusion, unlabeled_dataset: UnlabeledDataset) -> Dict[int, int]:
    pass


def has_converged(unlabeled_dataset: UnlabeledDataset):
    pass
