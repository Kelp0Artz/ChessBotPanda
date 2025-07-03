import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter('runs/exp4-after-v5') # Use tensorboard --logdir runs --port 6006 for visualization of training process
writer_interval = 50
checkpoint_interval = 1000
checkpoint_path = "../experiments_logs/CHECKPOINTS/checkpoint.pth"


import h5py
import chess
import numpy as np
np.set_printoptions(threshold=np.inf)

from chessEngine import ChessEngine 

# ----------------------------
#  Set device
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# ChessDataset Definition
# ----------------------------
class ChessDatasetSampler(Dataset, ChessEngine): # ADD a function, which will skip openings more, should help with overfitting
    """
    A dataset class for loading chess game data from an HDF5 file.
        self.map[key] = [game, scoreForWeight]
    SIZE OF BLOCK is the number of GAMES to load at once.
    """
    def __init__(self, dataset_path, size, size_of_block=50):
        super().__init__()
        self.file_path = dataset_path
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_size = file.attrs['totalNumGames']
        #self.dataset_size = size
        self.size_of_block = size_of_block
        self.MIN_NUM_SAMPLES_PER_GAME = 20
        [self.total_samples, self.total_samples_per_cache] = self.find_size_of_samples()
        self.cache = []
        self.current_block = None
    
    def find_size_of_samples(self):
        total_samples = 0
        total_samples_per_cache = []
        start = 0
        for x in range(0, (self.dataset_size // self.size_of_block) + 1):
            history = total_samples
            with h5py.File(self.file_path, 'r') as file:
                if 'games/moves' in file:
                    self.dataset = file['games/moves'][start:start + self.size_of_block]
                else:
                    print(f"Key 'games/moves' not found in {self.file_path}.")
                    return 0
            for idx, raw_game in enumerate(self.dataset):
                game = raw_game.decode('utf-8').split()
                if len(game) < self.MIN_NUM_SAMPLES_PER_GAME:
                    continue
                total_samples += len(game) - 1
            start += self.size_of_block 

            total_samples_per_cache.append(total_samples - history)
        print(f"Total samples found: {total_samples}")
        return total_samples, total_samples_per_cache
    
    def caching(self, block_idx):
        self.map = {} 
        self.game_keys = []

        start = block_idx * self.size_of_block
        end = start + self.size_of_block

        try:
            with h5py.File(self.file_path, 'r') as file:
                dataset = file['games/moves'][start:end]
                whoWon = file['games/whiteWon'][start:end]
                averageElo = file['games/averageElo'][start:end]
                sampleWeight = file['sampleWeightGroup/sampleWeight'][:]

        except Exception as e:
            print(f"File {self.file_path} wasn't found. Error: {e}")
            self.dataset = []

        for idx, raw_game in enumerate(dataset):
            game = raw_game.decode('utf-8').split()

            if len(game) < self.MIN_NUM_SAMPLES_PER_GAME:
                continue

            average_elo = averageElo[idx]
            scoreForWeight = sampleWeight[min(int(average_elo), 3200)-1]

            key = start + idx
            self.map[key] = [game, [scoreForWeight, average_elo, whoWon[idx]]] 
            self.game_keys.append(key)
            
        self.current_block = block_idx
    
    def create_one_hot_map(self, notation):
        """
        Create a one-hot encoded tensor for a given index and number of classes.
        Goes from 0 to 8x8x8x8-1 = 4095.
        from a to h, 1 to 8
        Squares from 0 to 63
        """
        promotions = {
            "e": 0,
            "r": 1,
            "n": 2,
            "b": 3,
            "q": 4
        }
        positions = {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "g": 7,
                "h": 8
            }
        if notation[-1] in promotions:
            promotion = promotions[notation[-1]]
            notation = notation[:-1]
        else: 
            promotion = 0
        index = 0
        count = 0
        from_square = None
        for char in notation:
            count += 1
            if char in positions:
                index = (positions[char]-1) * 8
            else:
                index += int(char) - 1
            if count == 2 and from_square == None:
                from_square = index
                count = 0
                index = 0
            else:
                to_square = index
        return from_square, to_square, promotion #return torch.nn.functional.one_hot(torch.tensor(from_square), 64), torch.nn.functional.one_hot(torch.tensor(to_square), 64), torch.nn.functional.one_hot(torch.tensor(promotion), 5) # r, n, b, q + empty
    
    def get_block_and_local_idx(self, global_idx):
        running_total = 0
        for block_idx, num_samples in enumerate(self.total_samples_per_cache):
            if global_idx < running_total + num_samples:
                local_idx = global_idx - running_total
                return block_idx, local_idx
            running_total += num_samples
        raise IndexError(f"Index {global_idx} is out of bounds. Total samples: {self.total_samples}")


    def __getitem__(self, idx):
        block_idx, local_idx = self.get_block_and_local_idx(idx)

        if block_idx != self.current_block:
            self.caching(block_idx)

        cumulative = 0
        for key in self.game_keys:
            game = self.map[key][0]
            num_samples = len(game) - 1

            if local_idx < cumulative + num_samples:
                scoreForWeight, average_elo, whoWon = self.map[key][1]
                local_sample_idx = local_idx - cumulative
                sample_length = local_sample_idx + 2
                sequence = game[:sample_length]

                static_state, _ = self.get_states(sequence)
                legal_moves = self.findLegalMove(sequence)
                target = self.create_one_hot_map(sequence[-1])
                sequence = [self.encode_move(move) for move in sequence]
                sequence = torch.tensor(sequence, dtype=torch.float32)

                return (sequence[0:-1], static_state[-2], legal_moves[-1], scoreForWeight, target, average_elo)

            cumulative += num_samples

        raise IndexError(f"Index {idx} is out of bounds for current block.")

    def __len__(self):
        return self.total_samples


# ----------------------------
# CHESSBot Definition
# ----------------------------
class CHESSBot(nn.Module):
    """
    A simple model that combines a CNN and a LSTM structure to predict the next move.
    Chosing this type of architecture was selected to capture both the spatial and temporal aspects of the game.
    Since the input is a sequence of moves, we use an LSTM to process the sequence.
    The board state is represented as an 8x8x8 tensor, which is processed by a CNN.
    We concatenate the output of LSTM and CNN, ranther than using CNN on LSTM ouput, which saved computation.
    """
    def __init__(self, dropout=0.1): 
        """
        Initializes the CHESSBot model.
        
        Archiutecture:
        - LSTM with 4 layers, hidden size of 128, and dropout of 0.3
        - CNN with 2 convolutional layers, each followed by batch normalization and max pooling
        - Fully connected layers with layer normalization and dropout
        - Output layers for predicting the from square, to square, and promotion type
        - The from square and to square outputs are 64-dimensional (for each square on the board)
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, num_layers=4, batch_first=True, dropout=0.3)
        self.ln_lstm = nn.LayerNorm(128)

        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 2 * 2 + 128 + 1, 1024) #The input is calculated with following variables CNN output + LSTM output + average elo
        self.ln1 = nn.LayerNorm(1024)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.drop2 = nn.Dropout(dropout)

        self.fc_from_square = nn.Linear(512,64)
        self.fc_to_square = nn.Linear(512, 64)
        self.fc_promotion = nn.Linear(512, 5)  #r, n, b, q + empty

    def forward(self, seq, board_state, average_elo, seq_length=None, temperature: float = 1.0): #Maybe add the actuall functionality of temperature you fucko
        """
        seq: [batch_size, seq_len, 4]
        board_state: [batch_size, 8, 8, 8] 
        tempreture: Controls the randomness of the output.
        """
        cnn_out = F.relu(self.bn1(self.conv1(board_state)))
        cnn_out = self.pool1(cnn_out)
        cnn_out = F.relu(self.bn2(self.conv2(cnn_out)))
        cnn_out = self.pool2(cnn_out)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        if seq_length is not None:
            seq_packed, _ = pack_padded_sequence(seq, seq_length.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(seq_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(seq)
        lstm_out = self.ln_lstm(lstm_out[:, -1, :])
        
        x = torch.cat([cnn_out, lstm_out, average_elo], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)

        from_square_logits = self.fc_from_square(x)
        to_square_logits = self.fc_to_square(x)
        promotion_logits = self.fc_promotion(x)

        return from_square_logits, to_square_logits, promotion_logits

# ----------------------------
# Checkpoint Save/Load Utils
# ----------------------------
def save_checkpoint(model, optimizer, epoch, loss, scaler, scheduler, path):
    """
    Save model, optimizer, epoch, loss, scaler, and scheduler states to a checkpoint file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

    print(f"Model saved at {path}\nEpoch: {epoch}\nLoss: {loss}")

def load_checkpoint(model, optimizer, scaler, scheduler, path):
    """
    Load model, optimizer, scaler, and scheduler states from a checkpoint file.
    """
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss  = checkpoint['loss']

    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        print("Warning: no 'scaler_state_dict' found in checkpoint; skipping AMP state.")

    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("Warning: no 'scheduler_state_dict' found in checkpoint; skipping scheduler state.")

    print(f"Model loaded from {path}\nEpoch: {epoch}\nLoss: {loss}")
    return model, optimizer, (epoch, loss, scaler, scheduler)

# ----------------------------
# Training Setup and Loop
# ----------------------------
def custom_collate(batch):
    """
    Custom collate function to handle variable-length sequences.
    ;d
    """
    seq, static_position, legal_moves, scoreForWeight, targets, average_elo = zip(*batch)
    seq = pad_sequence(seq, batch_first=True, padding_value=0)
    static_position = torch.tensor(np.stack(static_position), dtype=torch.float32)
    score = torch.tensor(np.stack(scoreForWeight), dtype=torch.float32)

    from_idxs, to_idxs, promotion_idxs = zip(*targets)
    from_idxs = torch.tensor(from_idxs, dtype=torch.long)   # [B,64]
    to_idxs = torch.tensor(to_idxs, dtype=torch.long)   # [B,64]
    promotion_idxs = torch.tensor(promotion_idxs, dtype=torch.long)   # [B,5]
    return seq, static_position, legal_moves, score, (from_idxs, to_idxs, promotion_idxs), average_elo

def main(): #ADD mixed precision
    # Dataset and DataLoader
    pathToDataset = r"..\dataset\sample.h5"
    dataset = ChessDatasetSampler(pathToDataset, 4000000)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.01 * len(dataset)) # Validation set is small, so it wouldn't take too long to compute
    test_size = len(dataset) - train_size - val_size

    # Use Subset instead of random_split to speed up the whole process of training
    from torch.utils.data import Subset
    train_indices = list(range(train_size))
    
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(dataset)))

     # Uncomment to see the sizes of the datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"Dataset sizes: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create DataLoader for each dataset
    train_loader = DataLoader(
        dataset, 
        batch_size=512, 
        shuffle=False, 
        collate_fn=custom_collate, 
        num_workers=8, 
        pin_memory=True
    )
    VALIDATION_INTERVAL = 5500

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
        pin_memory=False
    )

    # Model, Loss, Optimizer, and Scheduler
    model = CHESSBot().to(device)
    scaler = GradScaler() # mixed precision scaler

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000, verbose=True) # COMPLETE
    scheduler_check_interval = 1500 

    checkpoint_path = "../experiments_logs/CHECKPOINTS/ckpt_step_61000.pth"  # Change this to your desired checkpoint path
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model, optimizer, _ = load_checkpoint(model, optimizer, scaler, scheduler, checkpoint_path)
    else:
        print("\nNo checkpoint found. Starting training from scratch.\n")

    criterion_from_square = nn.CrossEntropyLoss(reduction="none")
    criterion_to_square = nn.CrossEntropyLoss(reduction="none")
    criterion_promotion = nn.CrossEntropyLoss(reduction="none")

    # Load weights from the dataset
    def get_weights():
        with h5py.File(pathToDataset, 'r') as file:
            return file['sampleWeightGroup']['sampleWeight'][()]
        
    weights = get_weights()
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    epochs = 100
    global_step = 0

    start_time = time.time()
    weighted_promotion_loss = 2.0 # CHANGE TO RECALL 

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_from = 0
        correct_to = 0
        correct_promo = 0
        ap_promo = 0
        scheduler_loss = 0.0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader): 
            #torch.cuda.reset_max_memory_allocated()   #later delete
            seq, static_position, legal_moves, scoreForWeight, (from_square_labels, to_square_labels, promotion_labels), average_elo = batch
        
            seq = seq.to(device)
            static_position = static_position.to(device)
            from_square_labels = from_square_labels.to(device)
            to_square_labels = to_square_labels.to(device)
            promotion_labels = promotion_labels.to(device)
            scoreForWeight = torch.tensor(scoreForWeight, dtype=torch.float32).to(device)
            average_elo = torch.tensor(average_elo, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()  

            with autocast(device_type=device.type):
                from_logits, to_logits, promo_logits = model(seq, static_position, average_elo, seq_length=None)
                loss_from = criterion_from_square(from_logits, from_square_labels)
                loss_to = criterion_to_square(to_logits, to_square_labels)
                loss_promo = criterion_promotion(promo_logits, promotion_labels)
                loss = (loss_from + loss_to + loss_promo * weighted_promotion_loss) * scoreForWeight
                weighted_loss = loss.mean()

            writer.add_scalar('Loss/train_batch', weighted_loss, global_step)

            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
              

            epoch_loss += weighted_loss.item()

            pred_from = from_logits.argmax(dim=1)
            pred_to = to_logits.argmax(dim=1)
            pred_promo = promo_logits.argmax(dim=1)

            

            correct_from += (pred_from == from_square_labels).sum().item()
            correct_to += (pred_to == to_square_labels).sum().item()
            correct_promo += ((pred_promo == promotion_labels) & (promotion_labels != 0)).sum().item()
            ap_promo += (promotion_labels != 0).sum().item()

            scheduler_loss += weighted_loss.item() * scoreForWeight.mean().item() # Variable for scheduler to adjust learning rate
            total += from_square_labels.size(0)

            if batch_idx % writer_interval == 0:
                time_spend_training = time.time() - start_time
                curr_acc_from  = correct_from  / total
                curr_acc_to    = correct_to    / total
                recall_promo   = correct_promo / (ap_promo + 1e-8)
                #curr_acc_promo = correct_promo / total
                writer.add_scalar('Accuracy/from_batch',    curr_acc_from,   global_step)
                writer.add_scalar('Accuracy/to_batch',      curr_acc_to,     global_step)
                writer.add_scalar('Recall/promo_batch',   recall_promo,  global_step)
                writer.add_scalar('Time/model', time_spend_training, global_step)
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {weighted_loss.item():.4f} | Acc_from: {curr_acc_from:.6f} | Acc_to: {curr_acc_to:.6f} | Acc_promo: {recall_promo:.6f} | Time: {time_spend_training:.2f}s")

            # Validate
            if global_step % VALIDATION_INTERVAL == 0 and batch_idx != 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=512,
                    shuffle=False,
                    collate_fn=custom_collate,
                    num_workers=0,
                    pin_memory=False
                )
                model.eval()
                val_loss = 0.0
                correct_from = 0
                correct_to = 0
                correct_promo = 0
                ap_promo = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        seq_val, static_position_val, legal_moves_val, scoreForWeight_val, (from_square_labels_val, to_square_labels_val, promotion_labels_val), average_elo_val = val_batch
                        seq_val = seq_val.to(device)
                        static_position_val = static_position_val.to(device)
                        from_square_labels_val = from_square_labels_val.to(device)
                        to_square_labels_val = to_square_labels_val.to(device)
                        promotion_labels_val = promotion_labels_val.to(device)
                        scoreForWeight_val = torch.tensor(scoreForWeight_val, dtype=torch.float32).to(device)
                        average_elo_val = torch.tensor(average_elo_val, dtype=torch.float32).unsqueeze(1).to(device)

                        with autocast(device_type=device.type):
                            from_logits_val, to_logits_val, promo_logits_val = model(seq_val, static_position_val, average_elo_val, seq_length=None)
                            pred_from_val = from_logits_val.argmax(dim=1)
                            pred_to_val = to_logits_val.argmax(dim=1)
                            pred_promo_val = promo_logits_val.argmax(dim=1)


                            correct_from_val += (pred_from_val == from_square_labels_val).sum().item()
                            correct_to_val += (pred_to_val == to_square_labels_val).sum().item()
                            correct_promo_val += (pred_promo_val == promotion_labels_val).sum().item()
                            ap_promo_val += (promotion_labels_val != 0).sum().item()

                            loss_from_val = criterion_from_square(from_logits_val, from_square_labels_val)
                            loss_to_val   = criterion_to_square(to_logits_val, to_square_labels_val)
                            loss_promo_val= criterion_promotion(promo_logits_val, promotion_labels_val)
                            loss = (loss_from_val + loss_to_val + loss_promo_val * weighted_promotion_loss) * scoreForWeight_val
                            val_loss += loss.mean().item()

                writer.add_scalar('Loss/validation', val_loss / len(val_loader), global_step)
                writer.add_scalar('Accuracy/from_validation', correct_from_val / len(val_loader.dataset), global_step)
                writer.add_scalar('Accuracy/to_validation', correct_to_val / len(val_loader.dataset), global_step)
                writer.add_scalar('Accuracy/promo_validation', correct_promo_val / ap_promo_val, global_step) 
                print(f"Validation Loss: {val_loss / len(val_loader):.4f} at step {global_step}")
                model.train()

            if global_step % scheduler_check_interval == 0:
                scheduler.step(scheduler_loss/ total)

            if global_step % checkpoint_interval == 0:
               ckpt_path = f'../experiments_logs/CHECKPOINTS/ckpt_step_{global_step}.pth'
               save_checkpoint(model, optimizer, epoch, weighted_loss.item(), scaler, scheduler, ckpt_path)
            
            global_step += 1

            #print(torch.cuda.max_memory_allocated() / 1e9, "GB used this batch") #later delete
            
        avg_loss = epoch_loss / len(train_loader)
        epoch_acc_from = correct_from / total
        epoch_acc_to   = correct_to   / total
        epoch_acc_promo= correct_promo / total

        writer.add_scalar('Loss/train_epoch',     avg_loss,       epoch+1)
        writer.add_scalar('Accuracy/from_epoch',  epoch_acc_from, epoch+1)
        writer.add_scalar('Accuracy/to_epoch',    epoch_acc_to,   epoch+1)
        writer.add_scalar('Accuracy/promo_epoch', epoch_acc_promo,epoch+1)
        writer.add_scalar('Learning_rate',        optimizer.param_groups[0]['lr'], epoch+1)
        

        print(f"Epoch {epoch+1} completed | Loss: {avg_loss:.4f} | Acc_from: {epoch_acc_from:.8f} | Acc_to: {epoch_acc_to:.8f} | Acc_promo: {epoch_acc_promo:.8f}")
        save_checkpoint(model, optimizer, epoch+1, avg_loss, f'../experiments_logs/CHECKPOINTS/checkpoint_epoch_{epoch+1}.pth')

        writer.close()
      
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  
    main()

