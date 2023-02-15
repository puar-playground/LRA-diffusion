import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import PIL
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class RVL_CDIP(data.Dataset):
    def __init__(self, data_root=None, split='train', cls_size=18976, l_max=None):
        self.data_root = data_root
        self.n_class = 16
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if split == 'train':
            file_path = os.path.join(self.data_root, 'labels/train.txt')
            label_path = os.path.join(self.data_root, 'labels/train.txt')

        elif split == 'val':
            file_path = os.path.join(self.data_root, 'labels/val.txt')
            label_path = os.path.join(self.data_root, 'labels/val.txt')

        else:
            file_path = os.path.join(self.data_root, 'labels/test.txt')
            label_path = os.path.join(self.data_root, 'labels/test.txt')

        with open(file_path) as fid:
            image_list = [os.path.join(line.split(' ')[0]) for line in fid.readlines()]

        with open(label_path) as fid:
            label_list = [int(line.split(' ')[1]) for line in fid.readlines()]

        if split != 'train':
            self.image_list = image_list
            self.label_list = label_list
        else:
            self.image_list = np.array(image_list[:l_max])
            self.label_list = torch.tensor(label_list[:l_max])

            l = np.array(self.label_list)
            x = np.unique(l)

            res_img_list = []
            res_label_list = []

            for i in x:
                idx = np.where(l == i)[0]
                idx = np.random.permutation(idx)
                idx = idx[:cls_size]

                res_img_list.append(self.image_list[idx])
                res_label_list.append(self.label_list[idx])

            self.image_list = np.concatenate(res_img_list).tolist()
            self.label_list = np.concatenate(res_label_list).tolist()

        self.targets = self.label_list  # this is for backward code compatibility

    def __getitem__(self, index):
        image_file_name = self.image_list[index]
        image_path = os.path.join(self.data_root, image_file_name)

        image = Image.open(image_path)
        image = image.resize((256, 256), resample=PIL.Image.BICUBIC)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        label = self.label_list[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.label_list)

    def update_corrupted_label(self, noise_label):
        self.label_list[:] = noise_label[:]
        self.targets = self.label_list


if __name__ == '__main__':
    data_dir = '/Users/chenjian/Desktop/working_space_LRA/LRA-diff_cifar/rvl-cdip'
    # get_train_labels(data_dir)
    # get_val_test_labels(data_dir)

    train_dataset = RVL_CDIP(data_root=data_dir, split='train')
    labels = train_dataset.targets
    print(len(labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True,
                                               num_workers=4, drop_last=True)

    for [img, l, i] in train_loader:
        print(img.shape)
        break
