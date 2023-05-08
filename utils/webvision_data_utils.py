import os
import random
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import PIL
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm


def resize(img, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        sw = sh = float(size) / size_min

        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        # sw = float(ow) / w
        # sh = float(oh) / h
    return img.resize((ow, oh), Image.BICUBIC)


class WebVision(data.Dataset):
    def __init__(self, data_root=None, split='train', balance=False, cls_size=500, randomize=False, transform='val'):
        self.data_root = data_root

        if transform == 'val':
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif transform == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            raise Exception('transform need to be train or val')

        if split == 'train':
            # flickr_path = os.path.join(self.data_root, 'info/train_filelist_flickr.txt')
            google_path = os.path.join(self.data_root, 'info/train_filelist_google.txt')

            # with open(flickr_path) as fid:
            #     list_flickr = np.array([line.strip().split(' ') for line in fid.readlines()])
            with open(google_path) as fid:
                list_google = np.array([[line.strip().split(' ')[0], line.strip().split(' ')[1]] for line in fid.readlines()
                                        if int(line.strip().split(' ')[1]) < 50])

            # image_list = list(list_flickr[:, 0]) + list(list_google[:, 0])
            image_list = list(list_google[:, 0])
            # label_list = list(list_flickr[:, 1]) + list(list_google[:, 1])
            label_list = [int(x) for x in list_google[:, 1]]

        elif split == 'val':
            file_path = os.path.join(self.data_root, 'info/val_filelist.txt')
            with open(file_path) as fid:
                list_val = np.array([['val_images_256/' + line.strip().split(' ')[0],
                                      line.strip().split(' ')[1]] for line in fid.readlines()
                                     if int(line.strip().split(' ')[1]) < 50])
            image_list = list(list_val[:, 0])
            label_list = [int(x) for x in list(list_val[:, 1])]

        else:
            raise Exception("split need to be train, val")


        if not balance:
            self.image_list = image_list
            self.label_list = label_list
        else:
            l = np.array(label_list)
            unique_labels = np.unique(l)
            min_class_cnt = np.min([np.sum(l == i) for i in unique_labels])

            if min_class_cnt < cls_size:
                cls_size = min_class_cnt
                print(f'sample not enough, use class size: {min_class_cnt}')

            self.image_list = np.array(image_list)
            self.label_list = torch.tensor(label_list)

            res_img_list = []
            res_label_list = []

            for i in unique_labels:
                idx = np.where(l == i)[0]

                if randomize:
                    idx = np.random.permutation(idx)

                idx = idx[:cls_size]
                res_img_list.append(self.image_list[idx])
                res_label_list.append(self.label_list[idx])

            self.image_list = np.concatenate(res_img_list).tolist()
            self.label_list = np.concatenate(res_label_list).tolist()
        
        self.targets = self.label_list  # this is for backward code compatibility

    def __getitem__(self, index):

        label = self.label_list[index]
        label = np.array(label).astype(np.int64)

        image_file_name = self.image_list[index]
        image_path = os.path.join(self.data_root, image_file_name)

        image = Image.open(image_path)
        image = resize(image, 256)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.label_list)

    def update_corrupted_label(self, noise_label):
        self.label_list[:] = noise_label[:]
        self.targets = self.label_list


if __name__ == '__main__':


    data_dir = './WebVision'

    train_dataset = WebVision(data_root=data_dir, split='val', randomize=False, balance=False)
    labels = train_dataset.targets
    print(len(labels))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True,
                                               num_workers=4, drop_last=True)

    with tqdm(enumerate(train_loader), total=len(train_loader), desc='train diffusion',
              ncols=120) as pbar:
        for i, (x_batch, y_batch, data_indices) in pbar:
            print(x_batch)
            print(y_batch)

            break