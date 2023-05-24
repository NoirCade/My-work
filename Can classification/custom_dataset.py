import glob
import os.path
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class customDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.all_path = glob.glob(os.path.join(root_path, '*', '*.png'))
        self.transform = transform
        self.label_dict = {}
        self.label_list = os.listdir(root_path)
        for i in range(len(self.label_list)):
            self.label_dict[self.label_list[i]] = int(i)

    def __getitem__(self, item):
        image_path = self.all_path[item]
        image = Image.open(image_path)
        label_name = image_path.split('\\')[1]
        label = self.label_dict[label_name]
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return len(self.all_path)


test = customDataset('./dataset/train')
for i in test:
    pass