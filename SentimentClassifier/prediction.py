import os, sys
import argparse
import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from alexnet import KitModel as AlexNet
from vgg19 import KitModel as VGG19


class ImageListDataset(Dataset):

    def __init__(self, list_filename, root=None, transform=None):
        super(ImageListDataset).__init__()

        with open(list_filename, 'r') as list_file:
            self.list = list(map(str.rstrip, list_file))

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        path = self.list[index]
        if self.root:
            path = os.path.join(self.root, path)

        x = default_loader(path)
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.list)


def sentiment_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
        transforms.ToTensor(),  # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the images color channels
    ])

def classify():
    model=AlexNet
    data = ImageListDataset(args.image_list, root=args.root, transform=transform)
    dataloader = DataLoader(data, batch_size=100, num_workers=8, pin_memory=True)
    with torch.no_grad():
        for x in tqdm(dataloader):
            p = model(x).numpy()  # order is (NEG, NEU, POS)
            #np.savetxt(sys.stdout.buffer, p, delimiter=',')