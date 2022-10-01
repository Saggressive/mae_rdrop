from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
from torchvision import transforms
# from numpy import random
import random
import os
import numpy as np
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
class Imagenet_Dataset(Dataset):
    def __init__(self, path, input_size):
        self.image_path = []
        images_folder_path=os.listdir(path)
        for folder in images_folder_path:
            folder_path=path+os.sep+folder
            folder_images_list=os.listdir(folder_path)
            images_path=[folder_path+os.sep+i for i in folder_images_list]
            self.image_path.extend(images_path)

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip()])

        self.transform_base = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        path = self.image_path[item]
        image = Image.open(path).convert("RGB")
        img=self.transform_train(image)
        img0,img1=self.transform_base(img),self.transform_aug(img)
        return img0,img1