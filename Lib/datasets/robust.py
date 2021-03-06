from torch.utils.data import Dataset
import torchvision.datasets as datasets
import os
import numpy as np
from PIL import Image
import torch

corruption = [
    "gaussian_noise.npy", "shot_noise.npy", "impulse_noise.npy", "defocus_blur.npy", "glass_blur.npy",
    "motion_blur.npy", "zoom_blur.npy", "snow.npy", "frost.npy", "fog.npy",
    "brightness.npy", "contrast.npy", "elastic_transform.npy", "pixelate.npy", "jpeg_compression.npy",
    "speckle_noise.npy", "gaussian_blur.npy", "spatter.npy", "saturate.npy",
]

corruption_imagenet = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def ImageNet_C_dataloader_generator(root, transform, bs):

    data_loader = []

    for c in corruption_imagenet:
        print(c)
        for s in range(1, 6):
            valdir = os.path.join(root, c, str(s))
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transform),
                batch_size=bs,
                shuffle=False,
                num_workers=16,
                pin_memory=True)

            data_loader.append((c, str(s), val_loader))

    return data_loader


class CIFAR10_C(Dataset):

    def __init__(self, root, transform, level):

        self.root = root
        self.transform = transform
        self.dataset_id = 0
        targets = np.load(os.path.join(self.root, "labels.npy"))
        self.targets = targets if level == -1 else targets[int(level*10000):int((level+1)*10000)]
        self.data_list = corruption
        assert -1 <= level <= 4
        self.level = level

        self.next_dataset()

    def next_dataset(self):

        if self.dataset_id <= len(self.data_list) - 1:
            data_name = self.data_list[self.dataset_id]
            data = np.load(os.path.join(self.root, data_name))
            self.data = data if self.level == -1 else data[int(self.level*10000):int((self.level+1)*10000)]
            self.data_name = data_name
            self.dataset_id += 1

        else:
            self.data = None
            self.data_name = None
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # target = torch.from_numpy(target).long()
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target.astype(np.long)


class CIFAR10_N_offline(CIFAR10_C):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.dataset_id = 0
        self.targets = np.load(os.path.join(self.root, "labels.npy"))
        self.data_list = ["std_0.05.npy", "std_0.1.npy", "std_0.2.npy"]
        self.next_dataset()


class CIFAR10_N_online(datasets.CIFAR10):

    def set_gaussion_param(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # add gaussion noise
        noised_img = np.clip(img / 255 + np.random.normal(size=img.shape, scale=self.std), 0., 1.) * 255
        img = noised_img.astype(np.uint8)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Tiny_Imagenet_N_online(datasets.ImageFolder):

    def set_gaussion_param(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        img = np.array(sample)
        # add gaussion noise
        noised_img = np.clip(img / 255 + np.random.normal(size=img.shape, scale=self.std), 0., 1.) * 255
        img = noised_img.astype(np.uint8)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        sample = Image.fromarray(img)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def save_cifar10_N(root, images, std):
    noised_img = np.clip(images / 255 + np.random.normal(size=images.shape, scale=std), 0., 1.) * 255
    img = noised_img.astype(np.uint8)

    os.makedirs(f"{root}/Gaussian_noise", exist_ok=True)
    np.save(f"{root}/Gaussian_noise/std_{std}.npy", img)


if __name__ == '__main__':

    data = 'cifar100'
    root = f'data/{data}/'

    if data == 'cifar10':
        cifar = datasets.CIFAR10(root=root, train=False, download=True)
    elif data == 'cifar100':
        cifar = datasets.CIFAR100(root=root, train=False, download=True)
    else:
        raise ValueError
    images = cifar.data  # (10000, 32, 32, 3)
    targets = cifar.targets

    # add gaussion noise
    # save noised images
    save_cifar10_N(root, images, 0.05)
    save_cifar10_N(root, images, 0.1)
    save_cifar10_N(root, images, 0.2)

    # save targets
    np.save(f"{root}/Gaussian_noise/labels.npy", targets)
