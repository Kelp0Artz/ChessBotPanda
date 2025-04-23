import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import h5py

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
        x = torch.tensor(self.dataX[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.dataY[idx], dtype=torch.long)
        return x, y

dataset = Dataset('dataset.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class ImageRecognition(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.CNN1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.CNN2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.CNN3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 128 * (input_size // 8) * (input_size // 8)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.CNN1(x)))
        x = self.pool(self.relu(self.CNN2(x)))
        x = self.pool(self.relu(self.CNN3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ImageRecognition(input_size=128, num_classes=13).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)

    prediction = model(data)
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    if i % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{i}.pth")
        print(f"Model saved at batch {i}")
