import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
from utils.ResNet_large import resnet50
from utils.clip_wrapper import clip_img_wrap
from utils.FashionCLIP_wrapper import FashionCLIP_wrap
from utils.cloth_data_utils import Clothing1M, get_train_labels, get_val_test_labels
import torch.optim as optim
from utils.learning import *
from model_diffusion import Diffusion
from utils.knn_utils import sample_knn_labels, knn, knn_labels, prepare_knn
import argparse
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)


def train(diffusion_model, val_loader, test_loader, device,
          model_save_dir, args, data_dir='./Clothing1M_data'):
    # device = diffusion_model.device

    k = args.k
    print(f'k:{k}')
    n_epochs = args.nepoch
    n_class = 14

    test_embed = np.load(os.path.join(data_dir, f'fp_embed_test_cloth_{args.fp_encoder}.npy'))
    val_embed = np.load(os.path.join(data_dir, f'fp_embed_val_cloth_{args.fp_encoder}.npy'))
    train_embed_all = torch.tensor(np.load(os.path.join(data_dir, f'fp_embed_train_cloth_{args.fp_encoder}.npy'))).to(
        torch.float32).to(device)
    train_labels = torch.tensor(np.load(os.path.join(data_dir, f'train_label_cloth.npy'))).to(device)

    # neighbours = np.load(os.path.join(data_dir, f'fp_knn_cloth_PLC.npy'))

    diffusion_model.fp_encoder.eval()
    params = list(diffusion_model.model.parameters()) + list(diffusion_model.diffusion_encoder.parameters())
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999),
                           amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')
    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    acc_diff = test(diffusion_model, test_loader, test_embed)
    print('test:', acc_diff)

    max_accuracy = 0.0
    print('Diffusion training start')
    for epoch in range(n_epochs):
        train_dataset = Clothing1M(data_root=data_dir, split='train', randomize=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, worker_init_fn=init_fn, drop_last=True)
        diffusion_model.diffusion_encoder.train()
        diffusion_model.model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, (x_batch, y_batch, data_indices) in pbar:

                with torch.no_grad():
                    fp_embd = diffusion_model.fp_encoder(x_batch.to(device))
                # fp_embd = train_embed_all[data_indices, :].to(device)

                y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch.to(device), train_embed_all,
                                                                  train_labels, k=k, n_class=n_class)
                # y_labels_batch, sample_weight = knn_labels(neighbours, data_indices, k=k, n_class=n_class)
                #
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

        acc_val = test(diffusion_model, val_loader, val_embed)
        if acc_val > max_accuracy:
            acc_test = test(diffusion_model, test_loader, test_embed)
            # save diffusion model
            # acc_test = test(diffusion_model, test_loader, test_embed)
            print(f"epoch: {epoch}, val accuracy: {acc_val:.2f}%, test accuracy: {acc_test:.2f}%")
            if args.device is None:
                states = [diffusion_model.model.module.state_dict(),
                          diffusion_model.diffusion_encoder.module.state_dict(),
                          diffusion_model.fp_encoder.module.state_dict()]
            else:
                states = [diffusion_model.model.state_dict(),
                          diffusion_model.diffusion_encoder.state_dict(),
                          diffusion_model.fp_encoder.state_dict()]
            torch.save(states, model_save_dir)
            print("Model saved, best val accuracy at Epoch {}.".format(epoch))
            max_accuracy = max(max_accuracy, acc_val)
        else:
            print(f"epoch: {epoch}, val accuracy: {acc_val:.2f}%")


def test(diffusion_model, test_loader, test_embed):

    if not torch.is_tensor(test_embed):
        test_embed = torch.tensor(test_embed).to(torch.float32)

    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        acc_avg = 0.
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'evaluating diff', ncols=100):
            [x_batch, target, indicies] = data_batch[:3]
            target = target.to(device)
            fp_embed = test_embed[indicies, :].to(device)
            label_t_0 = diffusion_model.reverse_ddim(x_batch, stochastic=False, fp_x=fp_embed).detach().cpu()
            acc_temp = accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
            acc_avg += acc_temp

        acc_avg /= len(test_loader)

    return acc_avg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--fp_encoder", default='PLC', help="encoder", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=1, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=2048, help="feature_dim", type=int)
    parser.add_argument("--k", default=120, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion", type=str)
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    args = parser.parse_args()

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    data_dir = os.path.join(os.getcwd(), 'Clothing1M_data')
    print('data_dir', data_dir)

    batch_size = args.batch_size
    num_workers = args.num_workers
    n_class = 14

    # prepare dataset directories
    get_train_labels(data_dir)
    get_val_test_labels(data_dir)

    # initialize diffusion model

    if args.fp_encoder == 'CLIP':
        train_dataset = Clothing1M(data_root=data_dir, split='train')
        val_dataset = Clothing1M(data_root=data_dir, split='val')
        test_dataset = Clothing1M(data_root=data_dir, split='test')
        fp_encoder_model = clip_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        fp_dim = fp_encoder_model.dim
    elif args.fp_encoder == 'FashionCLIP':
        train_dataset = Clothing1M(data_root=data_dir, split='train', Fashion=True)
        val_dataset = Clothing1M(data_root=data_dir, split='val', Fashion=True)
        test_dataset = Clothing1M(data_root=data_dir, split='test', Fashion=True)
        fp_encoder_model = FashionCLIP_wrap(device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        fp_dim = fp_encoder_model.dim
        # print(fp_dim)
    elif args.fp_encoder == 'PLC':
        train_dataset = Clothing1M(data_root=data_dir, split='train')
        val_dataset = Clothing1M(data_root=data_dir, split='val')
        test_dataset = Clothing1M(data_root=data_dir, split='test')
        fp_encoder_model = resnet50(num_classes=14).to(device)
        fp_dim = 14
        fp_state_dict = torch.load(os.path.join(os.getcwd(), 'model/PLC_Clothing1M.pt'), map_location=torch.device(device))
        fp_encoder_model.load_state_dict(fp_state_dict)
        fp_encoder_model.eval()

    train_labels = torch.tensor(train_dataset.targets)
    np.save(os.path.join(data_dir, f'train_label_cloth.npy'), train_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, worker_init_fn=init_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    model_path = './model/LRA-diffusion_Clothing1M_PLC.pt'
    diffusion_model = Diffusion(fp_encoder_model, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim, device=device,
                                feature_dim=args.feature_dim, encoder_type=args.diff_encoder,
                                ddim_num_steps=args.ddim_n_step, beta_schedule='cosine')
    state_dict = torch.load(model_path, map_location=torch.device(device))
    diffusion_model.load_diffusion_net(state_dict)

    # DataParallel wrapper
    diffusion_model.model = nn.DataParallel(diffusion_model.model).to(device)
    diffusion_model.diffusion_encoder = nn.DataParallel(diffusion_model.diffusion_encoder).to(device)
    diffusion_model.fp_encoder_model = nn.DataParallel(fp_encoder_model).to(device)

    # pre-compute for fp embeddings on training data
    print('pre-computing fp embeddings for training data')
    train_embed_dir = os.path.join(data_dir, f'fp_embed_train_cloth_{args.fp_encoder}.npy')
    train_embed = prepare_fp_x(diffusion_model.fp_encoder_model, train_dataset, train_embed_dir, device=device, fp_dim=fp_dim)
    # for validation data
    print('pre-computing fp embeddings for validation data')
    val_embed_dir = os.path.join(data_dir, f'fp_embed_val_cloth_{args.fp_encoder}.npy')
    val_embed = prepare_fp_x(diffusion_model.fp_encoder_model, val_dataset, val_embed_dir, device=device, fp_dim=fp_dim)
    # for testing data
    print('pre-computing fp embeddings for testing data')
    test_embed_dir = os.path.join(data_dir, f'fp_embed_test_cloth_{args.fp_encoder}.npy')
    test_embed = prepare_fp_x(diffusion_model.fp_encoder_model, test_dataset, test_embed_dir, device=device, fp_dim=fp_dim)

    # # pre-compute knns on training data
    # print('pre-compute knns on training data')
    # neighbours = prepare_knn(train_labels, train_embed,
    #                          os.path.join(data_dir, f'fp_knn_cloth_{args.fp_encoder}.npy'), k=args.k)

    # train the diffusion model
    train(diffusion_model, val_loader, test_loader, device, model_path, args, data_dir=data_dir)
    #


