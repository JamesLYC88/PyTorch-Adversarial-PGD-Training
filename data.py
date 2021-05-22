import os
import glob
from PIL import Image

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

cifar_10_mean = (0.491, 0.482, 0.447)
cifar_10_std = (0.202, 0.199, 0.201)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

def GetTrainLoader(batch_size):
	train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	return train_loader

def GetTestLoader(batch_size):
	test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
	return test_loader

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform=transform):
        self.images = []
        self.labels = []
        self.names = []
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)
