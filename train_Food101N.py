import os.path
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
from utils.clip_wrapper import clip_img_wrap
from utils.food_data_utils import Food101N, gen_train_list, gen_test_list
import torch
import torch.optim as optim
from utils.learning import cast_label_to_one_hot_and_prototype, adjust_learning_rate
from model_diffusion import Diffusion
from utils.knn_utils import knn_labels, knn
import argparse
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)


# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


def prepare_fp_x(fp_encoder, dataset, device, fp_dim=768):
    with torch.no_grad():
        data_loader = data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
        fp_embed_all = torch.zeros([len(dataset), fp_dim]).to(device)
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing embeddings fp(x)',
                  ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch, _, data_indecies] = data_batch[:3]
                temp = fp_encoder(x_batch.to(device))
                fp_embed_all[data_indecies, :] = temp

    return fp_embed_all


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])

    output = torch.softmax(-(output - 1)**2,  dim=-1)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def train(diffusion_model, train_loader, test_loader, model_save_dir, n_epochs=1000, knn=10, data_dir='./Food101N_data'):
    device = diffusion_model.device
    n_class = diffusion_model.n_class

    clip_test_embed = np.load(os.path.join(data_dir, 'fp_embed_test_food.npy'))
    clip_train_embed = np.load(os.path.join(data_dir, 'fp_embed_train_food.npy'))
    neighbours = np.load(os.path.join(data_dir, 'fp_knn_food.npy'))

    # acc_diff = test(diffusion_model, clip_test_embed, test_loader)
    # print('test:', acc_diff)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0.0
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, (x_batch, _, data_indices) in pbar:

                fp_embd = torch.tensor(clip_train_embed[data_indices, :]).to(torch.float32).to(device)
                y_labels_batch, sample_weight = knn_labels(neighbours, data_indices, k=knn, n_class=101)

                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch.to(device)

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=10, n_epochs=200, lr_input=0.001)
                n = x_batch.size(0)

                # antithetic sampling
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = diffusion_model.forward_t(y_0_batch, x_batch.to(device), t, fp_embd)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = torch.matmul(sample_weight.to(device), mse_loss)
                loss = torch.mean(weighted_mse_loss)

                pbar.set_postfix({'loss': loss.item()})

                # optimize diffusion model that predicts eps_theta
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        acc_test = test(diffusion_model, test_loader, clip_test_embed)
        print(f"epoch: {epoch}, diff accuracy: {acc_test:.2f}%")
        if acc_test > max_accuracy:
            # save diffusion model
            states = [diffusion_model.model.state_dict(), diffusion_model.fp_encoder.state_dict()]
            torch.save(states, model_save_dir)
            print("Model saved, update best accuracy at Epoch {}.".format(epoch))
            max_accuracy = max(max_accuracy, acc_test)


def test(diffusion_model, test_loader, test_embed):

    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        acc_avg = 0.
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'evaluating diff', ncols=100):
            [images, target, indicies] = data_batch[:3]
            target = target.to(device)

            fp_embed = torch.tensor(test_embed[indicies, :]).to(torch.float32).to(device)
            label_t_0 = diffusion_model.reverse_ddim(images, stochastic=False, fp_x=fp_embed).detach().cpu()
            acc_temp = accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
            acc_avg += acc_temp

        acc_avg /= len(test_loader)

    return acc_avg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=100, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=100, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=20, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=1024, help="feature_dim", type=int)
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet34_l',
                        help="which encoder for diffusion (resnet18_l, 34_l, 50_l...)", type=str)
    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), 'Food101N_data')

    # set device
    device = args.device
    print('Using device:', device)

    # load dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    n_class = 101

    # prepare dataset directories
    map_name2cat = gen_train_list(data_dir)
    gen_test_list(map_name2cat, data_dir)

    # create dataset and loader
    train_dataset = Food101N(data_path=data_dir, split='train')
    labels = torch.tensor(train_dataset.targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              worker_init_fn=_init_fn, drop_last=True)
    test_dataset = Food101N(data_path=data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # initialize diffusion model
    clip_model = clip_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    fp_dim = clip_model.dim
    model_path = './model/Food101N_clip_LRA-diffusion.pt'
    diffusion_model = Diffusion(clip_model, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim, device=device,
                                feature_dim=args.feature_dim, encoder_type=args.diff_encoder,
                                ddim_num_steps=args.ddim_n_step)
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # pre-compute for fp embeddings on training data
    print('pre-compute for fp embeddings on training data')
    train_embed = prepare_fp_x(clip_model, train_dataset, device=device, fp_dim=fp_dim)
    np.save(os.path.join(data_dir, 'fp_embed_train_food.npy'), train_embed.cpu())
    # for testing data
    print('pre-compute for fp embeddings on testing data')
    test_embed = prepare_fp_x(clip_model, test_dataset, device=device, fp_dim=fp_dim)
    np.save(os.path.join(data_dir, 'fp_embed_test_food.npy'), test_embed.cpu())

    # pre-compute knns on training data
    neighbours = torch.zeros([train_embed.shape[0], args.k + 1]).to(torch.long)
    for i in tqdm(range(int(train_embed.shape[0] / 100) + 1), desc='pre-compute knn for training data', ncols=100):
        start = i * 100
        end = min((i + 1) * 100, train_embed.shape[0])
        ebd = train_embed[start:end, :]
        _, neighbour_ind = knn(ebd, train_embed, k=args.k+1)
        neighbours[start:end, :] = labels[neighbour_ind]
    np.save(os.path.join(data_dir, 'fp_knn_food.npy'), neighbours)

    # train the diffusion model
    print('diffusion_knn_Food101N')
    train(diffusion_model, train_loader, test_loader, model_path, n_epochs=args.nepoch, knn=args.k, data_dir=data_dir)




