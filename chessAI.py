import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import h5py
import chess
import numpy as np
np.set_printoptions(threshold=np.inf)

# ----------------------------
#  Set device
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# ChessEngine Definition
# ----------------------------
class ChessEngine(): 
    """
    From Black to White
    A to H
    
    A basic chess engine to handle basic chess operations.
    Like:
    - encoding moves : self.encode_moves(self, moves)
    which takes a string of moves and returns a numpy array.
    - finding legal moves : self.findLegalMove(self, moves)
    which takes a string of moves and returns a list of numpy arrays.
    - get states : self.get_states(self, game)
    which takes a string of moves and returns a numpy array of board states.
    
    """
    def __init__(self):
        self.initial_board = np.array((
            [[0, 1, 0, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 1, 0]],

            [[1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 1, 0]],

            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],

            [[1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1]],

            [[0, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0, 0, 0, 1]]
        ))
        self.board = self.initial_board.copy() 
        self.promotions = {
            "r": 2,
            "n": 3,
            "b": 4,
            "q": 5
        }
        self.positions = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7
        }
    
    def board_reset(self):
        self.board = self.initial_board.copy()
        
    def move(self, value):
        SELECT = []
        POSITION = []
        promotion = False
        for char in value:
            if char in self.promotions and len(POSITION) == 2:
                promotion = self.promotions[char]
                continue
            if char in self.positions:
                char = int(self.positions[char])
            else:
                char = int(char)
                char -= 1
                char = -(char) - 1
            if len(SELECT) < 2:
                SELECT.append(char)
            else:
                POSITION.append(char)  
        if promotion:
            print('Promotion')
            self.board[POSITION[1]][POSITION[0]] = [0, 0, 0, 0, 0, 0, self.board[SELECT[1]][SELECT[0]][5], self.board[SELECT[1]][SELECT[0]][6]]
            self.board[POSITION[1]][POSITION[0]][promotion-1] = 1
        else:   
            self.board[POSITION[1]][POSITION[0]] = self.board[SELECT[1]][SELECT[0]]
        self.board[SELECT[1]][SELECT[0]] = [0, 0, 0, 0, 0, 0, 0, 0]

    def check_for_promotion(self, move):
        """
        Checks and reurns the promotion piece if the move is a promotion.
        Returns a value representing the promotion piece.
        """
        return self.promotions[move[4]] if len(move) == 5 else 0
    
    def get_states(self, game):
        """
        Get all the states of the game as a numpy array.
        Return a numpy array of a board representation in shape of [num_moves, 8, 8, 8].
        """
        arr = []
        self.board_reset()
        for move in game:
            self.move(move)
            arr.append(np.array(self.board)) 
        return np.array(arr)
    
    def encode_move(self, move):
        """
        Helper function to encode a single move string into a encoded format.
        """
        return [self.positions[move[0]], int(move[1]), self.positions[move[2]], int(move[3])]
    
    def findLegalMove(self, moves):
        """
        Compute legal moves for the given game history(list).
        Returns a list of numpy arrays (one per move) of encoded legal moves.
        z.B. input:["e2e4", "e7e5", "a2a4"]
        output: array([6, 8, 7, 6],.....)X3
        """
        board = chess.Board()
        legal_moves = []
        for move in moves:
            try:
                board.push_uci(move)
            except Exception as e:
                print(f"Invalid move {move}: {e}")
                continue
            encoded = [self.encode_move(m.uci()) for m in board.legal_moves]
            legal_moves.append(np.array(encoded))
        return legal_moves
engine = ChessEngine()
#print(engine.findLegalMove(["e2e4", "e7e5", "a2a4",]))

# ----------------------------
# ChessDataset Definition
# ----------------------------
class ChessDataset(Dataset, ChessEngine):
    """
    A dataset class for loading chess game data from an HDF5 file.
        self.map[key] = [game, scoreForWeight]
    """
    def __init__(self, dataset_path, size):
        super().__init__()
        self.file_path = dataset_path
        try:
            with h5py.File(self.file_path, 'r') as file:
                self.dataset = file['games/moves'][0:size]
                self.whoWon = file['games/whiteWon'][0:size]
                self.averageElo = file['games/averageElo'][0:size]
                self.sampleWeight = file['sampleWeightGroup/sampleWeight'][:]
        except Exception as e:
            print(f"File {self.file_path} wasn't found. Error: {e}")
            self.dataset = []

        self.sequences = []
        self.map = {}
        
        self.game_keys = []  
        self.total_samples = 0  

        key = 0
        for idx, raw_game in enumerate(self.dataset):
            
            game = raw_game.decode('utf-8').split()
            
            if len(game) < 2:
                continue
            scoreForWeight = self.sampleWeight[min(int(self.averageElo[idx]), 3200)-1]
            self.map[key] = [game, [scoreForWeight, self.whoWon[idx]]] 
            self.game_keys.append(key)
            
            self.total_samples += len(game) - 1
            key += 1
        
        
    def __getitem__(self, idx): 
        cumulative = 0  
        for key in self.game_keys:
            game = self.map[key][0]
            scoreForWeight = self.map[key][1][0]
            num_samples = len(game) - 1   
            if idx < cumulative + num_samples:
                
                local_sample_idx = idx - cumulative
                sample_length = local_sample_idx + 2  
                sequence = game[:sample_length]
                static_state = self.get_states(sequence)
                legal_moves = self.findLegalMove(sequence)
                sequence = [self.encode_move(move) for move in sequence]
                sequence = torch.tensor(sequence, dtype=torch.float32)
                return (sequence[0:-1], static_state[-2], legal_moves[-1], scoreForWeight, static_state[-1])
            cumulative += num_samples

        raise IndexError("Index out of range")
    
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
    def __init__(self): 
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=24, num_layers=3, batch_first=True)

        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 8 * 8 + 24, 3072)
        self.fc2 = nn.Linear(3072, 4116)

    def forward(self, seq, board_state, seq_length=None, temperature: float = 1.0):
        """
        seq: [batch_size, seq_len, 4]
        board_state: [batch_size, 8, 8, 8] 
        tempreture: Controls the randomness of the output.
        """
        cnn_out = torch.relu(self.conv1(board_state))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        if seq_length is not None:
            seq_packed = pack_padded_sequence(seq, seq_length.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(seq_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(seq)
        lstm_out = lstm_out[:, -1, :]
        
        x = torch.cat([cnn_out, lstm_out], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        if temperature != 1.0:
            x = x / temperature
        return x

# ----------------------------
# Checkpoint Save/Load Utils
# ----------------------------
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved at {path}\nEpoch: {epoch}\nLoss: {loss}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {path}\nEpoch: {epoch}\nLoss: {loss}")
    return model, optimizer

# ----------------------------
# Training Setup and Loop
# ----------------------------
def custom_collate(batch):
    """
    Custom collate function to handle variable-length sequences.
    ;d
    """
    seq, static_position, legal_moves, scoreForWeight, target = zip(*batch)
    seq = pad_sequence(seq, batch_first=True, padding_value=0)
    static_position = torch.tensor(np.stack(static_position), dtype=torch.float32)
    score = torch.tensor(np.stack(scoreForWeight), dtype=torch.float32)
    target = torch.tensor(np.stack(target).flatten(), dtype=torch.float32)
    return seq, static_position, legal_moves, score, target

# Dataset and DataLoader
pathToDataset = r"datasets\sample.h5"
dataset = ChessDataset(pathToDataset, 10)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=custom_collate) # Change to True for shuffling

# Model, Loss, Optimizer, and Scheduler
model = CHESSBot().to(device)
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Load weights from the dataset
def get_weights():
    with h5py.File(pathToDataset, 'r') as file:
        return file['sampleWeightGroup']['sampleWeight'][()]
    
weights = get_weights()
weights = torch.tensor(weights, dtype=torch.float32).to(device)

epochs = 100
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    # Use enumerate to track the batch index
    for batch_idx, batch in enumerate(dataloader): ###ADD TARGET VALUES SOMEHOW
        seq, static_position, legal_moves, scoreForWeight, target = batch
        print("ShAPE: ", static_position.shape)
        
        seq = seq.to(device)
        static_position = static_position.to(device)
        target = target.to(device)  

        predictions = model(seq, static_position).to(device)

        loss = criterion(predictions, target)
    
        weighted_loss = (loss * weights[scoreForWeight]).mean()

        weighted_loss.backward()  
        optimizer.step()   
        optimizer.zero_grad()    

        epoch_loss += weighted_loss.item()
        global_step += 1

    
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {weighted_loss.item():.4f}")

    scheduler.step()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
    
    # Optionally, save a checkpoint every 10 epochs.
    if (epoch+1) % 10 == 0:
        save_checkpoint(model, optimizer, epoch+1, avg_loss, f'checkpoint_epoch_{epoch+1}.pth')