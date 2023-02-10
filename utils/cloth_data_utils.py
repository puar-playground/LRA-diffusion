import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import PIL
import torchvision.transforms as transforms


class Clothing1M(data.Dataset):
    def __init__(self, data_root=None, split='train', cls_size=18976):
        self.data_root = data_root

        if split == 'train':
            file_path = os.path.join(self.data_root, 'annotations/noisy_train_key_list.txt')
            label_path = os.path.join(self.data_root, 'annotations/my_train_label.txt')
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        elif split == 'val':
            file_path = os.path.join(self.data_root, 'annotations/clean_val_key_list.txt')
            label_path = os.path.join(self.data_root, 'annotations/my_val_label.txt')
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            file_path = os.path.join(self.data_root, 'annotations/clean_test_key_list.txt')
            label_path = os.path.join(self.data_root, 'annotations/my_test_label.txt')
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

        with open(file_path) as fid:
            image_list = [line.strip() for line in fid.readlines()]

        with open(label_path) as fid:
            label_list = [int(line.strip()) for line in fid.readlines()]

        if split != 'train':
            self.image_list = image_list
            self.label_list = label_list
        else:
            self.image_list = np.array(image_list)
            self.label_list = np.array(label_list)

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


def get_train_labels(data_dir):
    train_file_list = data_dir + '/annotations/noisy_train_key_list.txt'
    noise_label_file = data_dir + '/annotations/noisy_label_kv.txt'

    # read train images
    fid = open(train_file_list)
    train_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(noise_label_file)
    label_list = [line.strip().split(' ') for line in fid.readlines()]

    label_map = dict()
    for m in label_list:
        label_map[m[0]] = m[1]

    train_labels = []
    for t in train_list:
        label = label_map[t]
        train_labels.append(label)

    print(len(train_labels))

    with open(data_dir + '/annotations/my_train_label.txt', 'w') as fid:
        for p in train_labels:
            fid.write('{}\n'.format(p))

    return train_labels


def get_val_test_labels(data_dir):
    val_file_list = data_dir + '/annotations/clean_val_key_list.txt'
    test_file_list = data_dir + '/annotations/clean_test_key_list.txt'
    clean_label_file = data_dir + '/annotations/clean_label_kv.txt'

    # read val images
    fid = open(val_file_list)
    val_list = [line.strip() for line in fid.readlines()]
    fid.close()

    # read test images
    fid = open(test_file_list)
    test_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(clean_label_file)
    label_list = [line.strip().split(' ') for line in fid.readlines()]
    fid.close()

    label_map = dict()
    for m in label_list:
        label_map[m[0]] = m[1]

    val_labels = []
    for t in val_list:
        label = label_map[t]
        val_labels.append(label)

    test_labels = []
    for t in test_list:
        label = label_map[t]
        test_labels.append(label)

    with open(data_dir + '/annotations/my_val_label.txt', 'w') as fid:
        for p in val_labels:
            fid.write('{}\n'.format(p))

    with open(data_dir + '/annotations/my_test_label.txt', 'w') as fid:
        for p in test_labels:
            fid.write('{}\n'.format(p))


if __name__ == '__main__':

    data_dir = '/Users/chenjian/Desktop/Label_noise_dataset/Clothing1M'
    # get_train_labels(data_dir)
    # get_val_test_labels(data_dir)

    train_dataset = Clothing1M(data_root=data_dir, split='train')
    labels = train_dataset.targets
    print(len(labels))