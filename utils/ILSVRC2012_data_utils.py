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


class ILSVRC2012(data.Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root

        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        file_path = os.path.join(self.data_root, 'ILSVRC2012_val_label.txt')
        with open(file_path) as fid:
            list_val = np.array([line.strip().split(' ') for line in fid.readlines() if int(line.strip().split(' ')[1]) < 50])

        self.image_list = ['ILSVRC2012_img_val/' + x for x in list_val[:, 0]]
        self.targets = [int(x) for x in list(list_val[:, 1])]

    def __getitem__(self, index):

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        image_file_name = self.image_list[index]
        image_path = os.path.join(self.data_root, image_file_name)

        image = Image.open(image_path)
        # image = resize(image, 256)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]



if __name__ == '__main__':


    data_dir = './ILSVRC2012/'
    train_dataset = ILSVRC2012(data_root=data_dir)
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