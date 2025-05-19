import os
import random
import re
import string
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models

chars = list(string.digits + string.ascii_letters)
chars.sort()
chars_len = len(chars)
max_capcha = 6


def encode(a):
    onehot = [0] * chars_len
    idx = chars.index(a)
    onehot[idx] += 1
    return onehot


class DatasetLoader(Dataset):

    # model size limit of image edit this if you want to change the size
    size = {
        "tiny": 62_500,
        "small": 125_000,
        "medium": 250_000,
        "large": None
    }

    def __init__(self, path, size, *, is_test=False, transform=None):
        self.transform = transform
        self.path = path
        data_images = os.listdir(self.path)
        random.shuffle(data_images)
        if self.size[size] is not None:
            data_images = data_images[:self.size[size]]

        test_len = len(data_images) // 10 if len(data_images) // 10 > 0 else 1
        train_len = len(data_images) - test_len
        if is_test:
            self.images = data_images[:test_len]
        else:
            self.images = data_images[:train_len]
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def get_name(self, idx):
        full_path = os.path.join(self.path, self.images[idx])
        return full_path, re.split(r"[\\/]", full_path)[-1].split(".")[0]

    def __getitem__(self, idx):
        img_path, label = self.get_name(idx)
        pil_img = Image.open(img_path).convert("L")
        lable_list = []
        for c in label:
            lable_list.append(encode(c))
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, np.array(lable_list), label


class OcrModel(torch.nn.Module):

    models = {
        "tiny": models.resnet18,
        "small": models.resnet34,
        "medium": models.resnet101,
        "large": models.resnet152
    }

    def __init__(self, size):
        super(OcrModel, self).__init__()
        self.model = self.models[size]()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(50, 160), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features=in_features, out_features=chars_len * max_capcha)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, max_capcha, chars_len)  # Reshape the output
        return x
