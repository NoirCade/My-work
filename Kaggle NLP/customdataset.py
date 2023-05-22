import pandas as pd
from torch.utils.data import Dataset, random_split
import torch

class MyDataset(Dataset):
    def init(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def len(self):
        return len(self.data)

    def getitem(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['target']
        return {'text': text, 'target': label}

    def train_val_split(self, train_size=0.9, random_state=42):
        dataset_size = len(self)
        train_size = int(train_size * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=torch.Generator().manual_seed(random_state))
        return train_dataset, val_dataset