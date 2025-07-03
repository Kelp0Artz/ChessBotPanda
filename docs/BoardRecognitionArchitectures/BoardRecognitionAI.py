import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import glob
import h5py
import os

from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter('experiment/recognition')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Dataset(torch.utils.data.Dataset):
    """
    Dataset for loading chess position images and their labels.

    Sample:
    
    """
    def __init__(self, data_path, noise_value=None): # ADD noise_value for data augmentation
        """
        Initializes the dataset with the path to the images and their labels.
        """
        super().__init__()
        self.datasetFolderPath = data_path
        self.images_paths = sorted(glob.glob(self.datasetFolderPath + "/*.*"),key=lambda x: int(''.join(filter(str.isdigit, x))))
        ###DIFFRENT NOTATION THAN IN THE OHTERS FILES
        self.positions = {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8
        }
        self.figures = {
            "e": 0, #e as empty square
            "p": 1,
            "r": 2,
            "n": 3,
            "b": 4,
            "k": 5,
            "q": 6
        }
        color = {
            "white": 1,
            "black": 0
        }

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx): #MAYBE ADD SEQUENCE
        image_path = self.images_paths[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        label = [self.figures[image_path[-7]], int(image_path[-5])]
        label = torch.tensor(label, dtype=torch.long)
        
        return (image, label)# maybe add seqeunce



class ImageRecognition(nn.Module):
    """
    A Convolutional Neural Network for image recognition of a chess position.
    The model predicts two outputs: figure type and color.
    The figure type is one of the 7 chess pieces (including empty square),
    and the color is either white or black.
    
    """
    def __init__(self, input_size, num_classes):
        """

        """
        super().__init__()
    
        self.relu = nn.ReLU()
        self.CNN1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.CNN2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.CNN3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 25088 #128 × 14 × 14 = 25088

        self.fc1 = nn.Linear(self.flattened_size, 4096)
        self.ln1 = nn.LayerNorm(4096)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(4096, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.drop2 = nn.Dropout(0.3)

        self.f3 = nn.Linear(1024, 128)
        self.ln3 = nn.LayerNorm(128)
        self.drop3 = nn.Dropout(0.3)

        self.fc_figure = nn.Linear(128, 7)  # for figure prediction
        self.fc_color = nn.Linear(128, 3)   # for color prediction

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.CNN1(x))))
        x = self.pool(self.bn2(F.relu(self.CNN2(x))))
        x = self.pool(self.bn3(F.relu(self.CNN3(x))))

        x = x.view(x.size(0), -1)
    
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = self.drop2(x)

        x = F.relu(self.f3(x))
        x = self.ln3(x)
        x = self.drop3(x)

        figure_logits = self.fc_figure(x)
        color_logits = self.fc_color(x)
        return figure_logits, color_logits

def main():
    dataset = Dataset(r'E:\Datasets\SOC\ChessPositionsRenders\cropped')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = ImageRecognition(input_size=115, num_classes=10).to(device)
    criterion_fig   = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    batch_interval = 10
    num_epochs = 50

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader, 1):
            #torch.cuda.reset_max_memory_allocated()
            data = data.to(device)
            target = target.to(device)

            fig_logits, col_logits = model(data)
            fig_labels = target[:,0]
            col_labels = target[:,1]

            loss_fig = criterion_fig(fig_logits, fig_labels)
            loss_col = criterion_color(col_logits, col_labels)
            loss = loss_fig + loss_col

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % batch_interval == 0:
                writer.add_scalar('Loss/train_batch', running_loss / batch_interval, epoch * len(train_loader) + i)
            #print(torch.cuda.max_memory_allocated() / 1e9, "GB used this batch")
        writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {running_loss/len(train_loader):.8f}")

        model.eval()
        val_loss = 0.0
        correct_fig = 0
        correct_col = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                fig_logits, col_logits = model(data)
                fig_labels = target[:,0]
                col_labels = target[:,1]

                loss_fig = criterion_fig(fig_logits, fig_labels)
                loss_col = criterion_color(col_logits, col_labels)
                loss = loss_fig + loss_col
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted_fig = torch.max(fig_logits, 1)
                _, predicted_col = torch.max(col_logits, 1)
                correct_fig += (predicted_fig == fig_labels).sum().item()
                correct_col += (predicted_col == col_labels).sum().item()
                total += fig_labels.size(0)

        writer.add_scalar('Loss/validation', val_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/figure', correct_fig/total, epoch)
        writer.add_scalar('Accuracy/color', correct_col/total, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss/len(val_loader):.8f}, "
            f"Figure Acc: {correct_fig/total:.8f}, Color Acc: {correct_col/total:.8f}")
        
        scheduler.step()
    MODEL_SAVE_PATH = r"E:\Datasets\SOC\ChessPositionsRenders\model.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support() 
    main()