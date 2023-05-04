import numpy as np
import torch
import torchvision
from utils.data_utils import *
from utils.clip_wrapper import clip_img_wrap
from utils.model_SimCLR import SimCLR_encoder
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
from tqdm import tqdm


def knn(query, data, train_labels, k=10):

    assert data.shape[1] == query.shape[1]

    # sim_matrix = torch.mm(query, data.t())

    sim_matrix = torch.cdist(query, data)
    # sim_weight, ind = sim_matrix.topk(k, dim=-1)

    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1, largest=False)

    # print(sim_indices[:10, :])

    sim_labels = train_labels[sim_indices].squeeze()
    # sim_labels = torch.gather(train_labels.expand(data.shape[0], -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / 0.5).exp()

    n_class = torch.unique(train_labels).shape[0]

    # counts for each class
    one_hot_label = torch.zeros(test_embed.shape[0] * k, n_class, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(test_embed.shape[0], -1, n_class) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)[:, :1].squeeze()

    return pred_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--k", default=10, help="num_workers", type=int)
    parser.add_argument("--data", default='cifar10', help="which dataset (cifar10 or cifar100)", type=str)
    parser.add_argument("--fp_encoder", default='SimCLR', help="which encoder (SimCLR or CLIP)", type=str)
    args = parser.parse_args()

    # load datasets
    if args.data == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='../', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='../', train=False, download=True)
        state_dict = torch.load('../model/SimCLR_128_cifar10.pt', map_location=torch.device(args.device))
    elif args.data == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='../', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='../', train=False, download=True)
    else:
        raise Exception("Date should be cifar10 or cifar100")

    # load fp encoder
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        state_dict = torch.load(f'../model/SimCLR_128_{args.data}.pt', map_location=torch.device(args.device))
        encoder_model = SimCLR_encoder(feature_dim=128).to(args.device)
        encoder_model.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'CLIP':
        encoder_model = clip_img_wrap('ViT-L/14', args.device)
        fp_dim = encoder_model.dim
    else:
        raise Exception("fp_encoder should be SimCLR or CLIP")

    train_dataset = Custom_dataset(train_dataset_cifar.data[:45000], train_dataset_cifar.targets[:45000])
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

    train_labels = torch.tensor(train_dataset.targets).squeeze().to(args.device)
    test_labels = torch.tensor(test_dataset.targets).squeeze().to(args.device)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    # compute embedding fp(x) for training and testing set
    with torch.no_grad():

        train_embed = []
        for i, (images, labels, indices) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=150):
            train_embed.append(encoder_model(images.to(args.device)))

        train_embed = torch.cat(train_embed, dim=0).contiguous()

        test_embed = []
        for i, (images, labels, indices) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=150):
            feature = encoder_model(images.to(args.device))
            test_embed.append(feature)
        test_embed = torch.cat(test_embed, dim=0).contiguous()

    pred_labels = knn(test_embed, train_embed, train_labels, k=args.k)
    acc = torch.sum(pred_labels == test_labels)
    acc = acc / pred_labels.shape[0]
    print(f'KNN accuracy: {100 * acc:.2f}, feature space: {args.fp_encoder}')