import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import h5py
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.dataset_path = data_path
        with h5py.File(self.dataset_path, 'r') as f:
            self.data_size = f.attrs['data_size']
            self.dataX = f['dataX'][:]
            self.dataY = f['dataY'][:]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return torch.tensor(self.dataX[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.dataY[idx], dtype=torch.long)

dataset = Dataset('dataset.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



class RecognitionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(RecognitionModel, self).__init__()
        self.CNN1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.CNN2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.CNN3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(128 * (input_size // 8) * (input_size // 8), 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x= self.CNN1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.CNN2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.CNN3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

for i, (data, target) in enumerate(dataloader):
    