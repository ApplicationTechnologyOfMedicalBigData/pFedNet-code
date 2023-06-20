import os
import nibabel as nib
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np
from skimage.transform import resize
from scipy import ndimage
from config import *
from tqdm import tqdm
import random
import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning)


def read_nifti_cls_file(filepath, rotate=False):
    # Read file
    img = nib.load(filepath)
    # Get raw data H W D C
    img = img.get_fdata().astype(np.float32)
    dim = len(img.shape)
    if dim == 3:
        if rotate:
            img = ndimage.rotate(img, 90, reshape=False)
            img = resize(img, (img_size[0], img_size[1], img_size[2]), order=0)
            img = np.array(img)
        else:
            img = resize(img, (img_size[0], img_size[1], img_size[2]), order=0)
            img = np.array(img)
    if dim == 4:
        img = resize(img, (img_size[0], img_size[1], img_size[2], img.shape[3]), order=0)
        img = np.array(img)
    if np.min(img) < np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)

    return img

def read_nifti_seg_file(filepath):
    # Read file
    img = nib.load(filepath)
    # img = img.get_fdata().astype(np.float32)

    # Get raw data H W D C
    img = img.get_fdata().astype(np.float32)
    # nib.viewers.OrthoSlicer3D(img).show()
    raw_shape = img.shape
    if len(raw_shape) == 3:
        img = torch.tensor(img,dtype=float,device=DEVICE)
        x = img
        y = x

        # 对背景外的区域进行归一化
        x -= y.mean()
        # std=y.std()
        x /= y.std()
        img = x


    if len(raw_shape) == 4:
        # H W D C -> C H W D
        img = torch.tensor(img,dtype=float,device=DEVICE).reshape(raw_shape[3],raw_shape[0],raw_shape[1],raw_shape[2])
        mask = torch.sum(img,dim=0) > 0
        for k in range(img.shape[0]):
            x = img[k,...]
            y = x[mask]

            # 对背景外的区域进行归一化
            x[mask] -= y.mean()
            # std=y.std()
            x[mask] /= y.std()
            img[k, ...] = x
    img=img.cpu().detach().numpy()

    return img


def get_cls_label(file_path):
    label_dir, file_name = os.path.split(file_path)
    hospital_dir, label = os.path.split(label_dir)

    return label


def get_nifti_seg_label(file_path,by_torch=False):
    label = nib.load(file_path)
    label = label.get_fdata().astype(np.int64)
    if "2D" in model:
        label=label[0]
    # nib.viewers.OrthoSlicer3D(label).show()
    # if by_torch==True:
    #     label=torch.tensor(label,dtype=int)
    return label


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x,k) for x in image],axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        (c, h, w, d) = image.shape
        h1 = np.random.randint(0, h - self.output_size[0])
        w1 = np.random.randint(0, w - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return {'image': image, 'label': label}


class RGBClsImage(Dataset):
    def __init__(self, srcpath):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
        self.srcpath = srcpath
        self.dataset = {"image": [], "label": []}
        # print("<<<<<<<<<<<<<<<<<<<<Waiting For Data Loading<<<<<<<<<<<<<<<<<<<<" + "\n")
        for roots, dirs, files in os.walk(self.srcpath):
            for file in files:
                file_path = os.path.join(roots, file)

                #########################################
                #the cvs is error
                if re.findall("csv",file_path):
                    continue
                ##########################################

                label = int(get_cls_label(file_path))
                image = cv2.imread(file_path)
                image = cv2.resize(image, (img_size[0], img_size[1]))
                image = self.transform(image)
                self.dataset['image'].append(image)

                self.dataset['label'].append(label)
        self.dataset['label'] = torch.LongTensor(self.dataset['label'])
        # print("<<<<<<<<<<<<<<<<<<<<Data Loading Complete<<<<<<<<<<<<<<<<<<<<" + "\n")

    def __getitem__(self, index):
        image, label = self.dataset['image'][index], self.dataset['label'][index]

        return image, label

    def __len__(self):
        return len(self.dataset['image'])


class CTClsDataset(Dataset):
    def __init__(self, srcpath):
        self.srcpath = srcpath
        self.dataset ={"image": [], "label": []}
        self.transform = transforms.ToTensor()
        print("<<<<<<<<<<<<<<<<<<<<Waiting For Data Loading<<<<<<<<<<<<<<<<<<<<" + "\n")
        for roots, dirs, files in os.walk(self.srcpath):
            for file in tqdm(files):
                file_path = os.path.join(roots, file)
                label = int(get_cls_label(file_path))
                image = read_nifti_cls_file(file_path, rotate)

                if len(image.shape) == 3:
                    # 增加一维channel，确保tensor输入为NCDHW
                    image = np.expand_dims(image, axis=0)
                    image = torch.from_numpy(image)
                    image = torch.permute(image, (0, 3, 1, 2))
                else:
                    image = image.astype(np.float32)
                    image = torch.from_numpy(image)
                    image = torch.permute(image, (3, 2, 0, 1))

                self.dataset['image'].append(image)
                self.dataset['label'].append(label)

        self.dataset["label"] = torch.LongTensor(self.dataset['label'])
        print("<<<<<<<<<<<<<<<<<<<<Data Loading Complete<<<<<<<<<<<<<<<<<<<<" + "\n")

    def __getitem__(self, index):
        image, label = self.dataset['image'][index], self.dataset['label'][index]

        return image, label

    def __len__(self):
        return len(self.dataset['image'])


class RGBSegImage(Dataset):
    def __init__(self, srcpath):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
        self.srcpath = srcpath
        self.dataset = {"image": [], "label": []}
        imgpath = os.path.join(self.srcpath, "images")
        labelpath = os.path.join(self.srcpath, "labels")
        print("<<<<<<<<<<<<<<<<<<<<Waiting For Images Loading<<<<<<<<<<<<<<<<<<<<" + "\n")
        for img_name in tqdm(os.listdir(imgpath)):
            file_path = os.path.join(imgpath, img_name)
            image = cv2.imread(file_path)
            image = cv2.resize(image, (img_size[0], img_size[1]))
            image = self.transform(image)
            self.dataset['image'].append(image)
        print("<<<<<<<<<<<<<<<<<<<<Waiting For Labels Loading<<<<<<<<<<<<<<<<<<<<" + "\n")
        for label_name in tqdm(os.listdir(labelpath)):
            label_path = os.path.join(labelpath, label_name)
            label = cv2.imread(label_path)
            label = cv2.resize(label, (img_size[0], img_size[1]))
            label = torch.from_numpy(label)
            self.dataset['label'].append(label)

        print("<<<<<<<<<<<<<<<<<<<<Data Loading Complete<<<<<<<<<<<<<<<<<<<<" + "\n")

    def __getitem__(self, index):
        image, label = self.dataset['image'][index], self.dataset['label'][index]

        return image, label

    def __len__(self):
        return len(self.dataset['image'])


class CTSegDataset(Dataset):
    def __init__(self, srcpath, transform=None):
        self.srcpath = srcpath
        imgpath = os.path.join(self.srcpath, "images")
        labelpath = os.path.join(self.srcpath, "labels")
        # print("<<<<<<<<<<<<<<<<<<<<Waiting For Images Loading<<<<<<<<<<<<<<<<<<<<" + "\n")
        self.img_list = [read_nifti_seg_file(os.path.join(imgpath, img_name)) for img_name in tqdm(os.listdir(imgpath))]
        # print("<<<<<<<<<<<<<<<<<<<<Waiting For Labels Loading<<<<<<<<<<<<<<<<<<<<" + "\n")
        self.label_list = [get_nifti_seg_label(os.path.join(labelpath, label_name)) for label_name in tqdm(os.listdir(labelpath))]
            # print("<<<<<<<<<<<<<<<<<<<<Data Loading Complete<<<<<<<<<<<<<<<<<<<<" + "\n")
        self.transform = transform

    def __getitem__(self, index):
        # 加载图像
        image = self.img_list[index]
        label = self.label_list[index]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.img_list)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


def data_loader(train_dataset, test_dataset):
    trainloader = data.DataLoader(train_dataset, batch_size, num_workers=0)
    testloader = data.DataLoader(test_dataset, batch_size, num_workers=0)
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    return trainloader, testloader, num_examples


def random_split_data_loader(dataset):
    data_len = len(dataset)
    train_size = int(data_len * 0.8)
    test_size = data_len - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader, testloader, num_examples = data_loader(trainset, testset)
    return trainloader, testloader, num_examples


def load_random_split_data(srcpath):

    if task == "classification" and data_type == "CT":
        dataset = CTClsDataset(srcpath)
        trainloader, testloader, num_examples = random_split_data_loader(dataset)

        return trainloader, testloader, num_examples

    if task == "classification" and data_type == "RGB":
        dataset = RGBClsImage(srcpath)
        trainloader, testloader, num_examples = random_split_data_loader(dataset)

        return trainloader, testloader, num_examples

    if task == "segmentation" and data_type == "CT":
        dataset = CTSegDataset(srcpath, transform=transforms.Compose([
        # RandomRotFlip(),
        # RandomCrop((img_size[0], img_size[1], img_size[2])),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
        trainloader, testloader, num_examples = random_split_data_loader(dataset)
        

        return trainloader, testloader, num_examples


def load_data_from_path(train_path, test_path):

    if task == "classification" and data_type == "CT":
        train_dataset = CTClsDataset(train_path)
        test_dataset = CTClsDataset(test_path)
        trainloader, testloader, num_examples = data_loader(train_dataset, test_dataset)

        return trainloader, testloader, num_examples

    if task == "classification" and data_type == "RGB":
        train_dataset = RGBClsImage(train_path)
        test_dataset = RGBClsImage(test_path)
        trainloader, testloader, num_examples = data_loader(train_dataset, test_dataset)

        return trainloader, testloader, num_examples

    if task == "segmentation" and data_type == "CT":
        train_dataset = CTSegDataset(train_path, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop((img_size[0], img_size[1], img_size[2])),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
        test_dataset = CTSegDataset(test_path, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop((img_size[0], img_size[1], img_size[2])),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
        trainloader, testloader, num_examples = data_loader(train_dataset, test_dataset)

        return trainloader, testloader, num_examples
