import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
import os
from utils.ResNet_large import resnet50
from utils.clip_wrapper import clip_img_wrap
from utils.cloth_data_utils import Clothing1M, get_train_labels, get_val_test_labels
import torch
import torch.optim as optim
from utils.learning import cast_label_to_one_hot_and_prototype, adjust_learning_rate
from model_diffusion import Diffusion
from utils.knn_utils import sample_knn_labels, knn, knn_labels
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


def train(diffusion_model, train_loader, val_loader, test_loader, model_save_dir, n_epochs=1000, knn=10, data_dir='./Clothing1M_data'):
    device = diffusion_model.device
    n_class = diffusion_model.n_class

    test_embed = np.load(os.path.join(data_dir, 'fp_embed_test_cloth.npy'))
    val_embed = np.load(os.path.join(data_dir, 'fp_embed_val_cloth.npy'))
    train_embed_all = np.load(os.path.join(data_dir, 'fp_embed_train_cloth.npy'))
    neighbours = np.load(os.path.join(data_dir, 'fp_knn_cloth.npy'))

    # acc_diff = test(diffusion_model, test_embed, test_loader)
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
            for i, (x_batch, y_batch, data_indices) in pbar:

                fp_embd = diffusion_model.fp_encoder(x_batch.to(device))
                # fp_embd = torch.tensor(clip_train_embed[data_indices, :]).to(torch.float32).to(device)
                # y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch.to(device), train_embed_all,
                #                                                   torch.tensor(train_dataset.targets).to(device),
                #                                                   k=knn, n_class=n_class)

                y_labels_batch, sample_weight = knn_labels(neighbours, data_indices, k=knn, n_class=n_class)

                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch.to(device)

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=1, n_epochs=n_epochs, lr_input=0.001)
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

        acc_val = test(diffusion_model, val_embed, val_loader)
        if acc_val > max_accuracy:
            # save diffusion model
            acc_test = test(diffusion_model, test_embed, test_loader)
            print(f"epoch: {epoch}, val accuracy: {acc_val:.2f}%, test accuracy: {acc_test:.2f}%")
            states = [diffusion_model.model.state_dict(), diffusion_model.fp_encoder.state_dict()]
            torch.save(states, model_save_dir)
            print("Model saved, best val accuracy at Epoch {}.".format(epoch))
            max_accuracy = max(max_accuracy, acc_val)
        else:
            print(f"epoch: {epoch}, val accuracy: {acc_val:.2f}%")


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
    print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=100, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=50, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--fp_encoder", default='PLC', help="encoder", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=20, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=2048, help="feature_dim", type=int)
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--data_dir", default='/home/jian/Clothing1M', help="dir to Clothing1M", type=str)
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion", type=str)
    args = parser.parse_args()

    # data_dir = os.path.join(os.getcwd(), 'Clothing1M_data')
    data_dir = args.data_dir

    # set device
    device = args.device
    print('Using device:', device)

    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    n_class = 14

    # prepare dataset directories
    get_train_labels(data_dir)
    get_val_test_labels(data_dir)

    train_dataset = Clothing1M(data_root=data_dir, split='train')
    labels = torch.tensor(train_dataset.targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers,
                                              worker_init_fn=_init_fn, drop_last=True)
    val_dataset = Clothing1M(data_root=data_dir, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = Clothing1M(data_root=data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # initialize diffusion model

    if args.fp_encoder == 'CLIP':
        fp_encoder_model = clip_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        fp_dim = fp_encoder_model.dim
    elif args.fp_encoder == 'PLC':
        fp_encoder_model = resnet50(num_classes=14).to(device)
        fp_dim = 14
        fp_state_dict = torch.load(os.path.join(os.getcwd(), 'model/PLC_Clothing1M.pt'), map_location=torch.device(device))
        fp_encoder_model.load_state_dict(fp_state_dict)
        fp_encoder_model.eval()

    model_path = './model/Clothing1M_LRA-diffusion.pt'
    diffusion_model = Diffusion(fp_encoder_model, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim, device=device,
                                feature_dim=args.feature_dim, encoder_type=args.diff_encoder,
                                ddim_num_steps=args.ddim_n_step)
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # pre-compute for fp embeddings on training data
    print('pre-compute for fp embeddings on training data')
    train_embed = prepare_fp_x(fp_encoder_model, train_dataset, device=device, fp_dim=fp_dim)
    np.save(os.path.join(data_dir, 'fp_embed_train_cloth.npy'), train_embed.cpu())
    # for validation data
    print('pre-compute for fp embeddings on validation data')
    val_embed = prepare_fp_x(fp_encoder_model, val_dataset, device=device, fp_dim=fp_dim)
    np.save(os.path.join(data_dir, 'fp_embed_val_cloth.npy'), val_embed.cpu())
    # for testing data
    print('pre-compute for fp embeddings on testing data')
    test_embed = prepare_fp_x(fp_encoder_model, test_dataset, device=device, fp_dim=fp_dim)
    np.save(os.path.join(data_dir, 'fp_embed_test_cloth.npy'), test_embed.cpu())

    # pre-compute knns on training data
    neighbours = torch.zeros([train_embed.shape[0], args.k + 1]).to(torch.long)
    for i in tqdm(range(int(train_embed.shape[0] / 100) + 1), desc='pre-compute knn for training data', ncols=100):
        start = i * 100
        end = min((i + 1) * 100, train_embed.shape[0])
        ebd = train_embed[start:end, :]
        _, neighbour_ind = knn(ebd, train_embed, k=args.k + 1)
        neighbours[start:end, :] = labels[neighbour_ind]
    np.save(os.path.join(data_dir, 'fp_knn_cloth.npy'), neighbours)

    # train the diffusion model
    print('diffusion_knn_Clothing1M')
    train(diffusion_model, train_loader, val_loader, test_loader, model_path, n_epochs=args.nepoch, knn=args.k, data_dir=data_dir)



