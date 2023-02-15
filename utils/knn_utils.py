import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

def knn(query, data, k=10):

    assert data.shape[1] == query.shape[1]

    M = torch.cdist(query, data)
    # M = 1 - torch.mm(query, data.t())
    v, ind = M.topk(k, largest=False)

    return v, ind[:, 0:min(k, data.shape[0])].to(torch.long)


def sample_knn_labels(query_embd, y_query, prior_embd, labels, k=10, n_class=10, weighted=False):

    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), torch.randint(0, k+1, (n_sample,))]

    # convert labels to bincount (row wise)
    y_one_hot_batch = nn.functional.one_hot(neighbour_label_distribution, num_classes=n_class).float()

    # max_agree, _ = torch.max(torch.sum(y_one_hot_batch, dim=1), dim=1)

    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]

    # normalize max count as weight
    if weighted:
        weights = neighbour_freq / torch.sum(neighbour_freq)
    else:
        weights = 1/ n_sample * torch.ones([n_sample]).to(query_embd.device)

    return sampled_labels, torch.squeeze(weights)


def knn_labels(neighbours, indices, k=5, n_class=101):

    n_sample = len(indices)

    # compute the label of nearest neighbours
    neighbour_label_distribution = torch.tensor(neighbours[indices, :k+1]).to(torch.long)

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), torch.randint(0, k+1, (n_sample,))]

    # convert labels to bincount (row wise)
    y_one_hot_batch = nn.functional.one_hot(neighbour_label_distribution, num_classes=n_class).float()

    # max_agree, _ = torch.max(torch.sum(y_one_hot_batch, dim=1), dim=1)

    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]

    # normalize max count as weight
    weights = neighbour_freq / torch.sum(neighbour_freq)

    return sampled_labels, torch.squeeze(weights)


def prepare_knn(labels, train_embed, save_dir, k=10):

    if os.path.exists(save_dir):
        neighbours = torch.tensor(np.load(save_dir))
        print(f'knn were computed before, loaded from: {save_dir}')
    else:
        neighbours = torch.zeros([train_embed.shape[0], k + 1]).to(torch.long)
        for i in tqdm(range(int(train_embed.shape[0] / 100) + 1), desc='pre-compute knn for training data', ncols=100):
            start = i * 100
            end = min((i + 1) * 100, train_embed.shape[0])
            ebd = train_embed[start:end, :]
            _, neighbour_ind = knn(ebd, train_embed, k=k + 1)
            neighbours[start:end, :] = labels[neighbour_ind]
        np.save(save_dir, neighbours)

    return neighbours