import torch 
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import h5py
import chess
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

class ChessEngine():
    """
    Which one is A or B?
    <<From Right to Left<<
    [PIECE, COLOR, WHICH ONE]
    self.piece = {
        "Pawn": 1,
        "Rook": 2,
        "Knight": 3,
        "Bishop": 4,
        "Queen": 5,
        "King": 6
    }
    self.color = {
        "white": 2,
        "black": 1 #IDK which one
    }
    """
    def __init__(self):
        #self.board = np.zeros((8,8,8))  #8x8xpiece(6)+color(2) = 512
        self.board = np.array((
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
        self.promotions =  {
            "r": 2,
            "n": 3,
            "b" : 4,
            "q" : 5
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
        self.board = np.array((
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
                char = -(char)-1
            if len(SELECT) < 2:
                SELECT.append(char)
            else:
                POSITION.append(char)  
        if promotion != False:
            print('Promotion')
            self.board[POSITION[1]][POSITION[0]] = [0,0,0,0,0,0, self.board[SELECT[1]][SELECT[0]][5], self.board[SELECT[1]][SELECT[0]][6]]
            self.board[POSITION[1]][POSITION[0]][promotion-1] = 1
        else:   
            self.board[POSITION[1]][POSITION[0]] = self.board[SELECT[1]][SELECT[0]]
        self.board[SELECT[1]][SELECT[0]] = [0, 0, 0, 0, 0, 0, 0, 0]  

    def get_states(self, game):
        """
        Get all the states of the game
        """
        array = []
        for move in game:
            self.move(move)
            array.append(self.board)
        return np.array(array)
    
    def encode_moves(self, moves):
        """
        Encode moves for neural network
        input should be a list of moves
        """
        encoded_moves = []
        for move in moves:
            encoded_moves.append([self.positions[move[0]], int(move[1]), self.positions[move[2]], int(move[3])])
        return np.array(encoded_moves)
    
    def findLegalMove(self, game):
        board = chess.Board()
        legal_moves = []
        for move in game:
            board.push_san(move)
            legal_move = [board.uci(move) for move in board.legal_moves]
            legal_move = list(map(self.encode_move, legal_move))
            legal_moves.append(np.array(legal_move))
        return legal_moves
    
class ChessDataset(Dataset, ChessEngine):
    def __init__(self, pathFile):
        """
        current_data_position: [game, move in game]
        Saves the current position of the data, for the next iteration.
        """
        super().__init__()
        self.pathFile = pathFile
        self.games = h5py.File(self.pathFile, 'r')['games']
        self.weights = h5py.File(self.pathFile, 'r')['sampleWeightGroup']['sampleWeight']
        self.current_data_position = [1, 0] #game, move in game
        self.current_serie_moves = self.games[f"game-{self.current_data_position[0]}"]['moves'][()].decode('utf-8').split()

        with h5py.File(self.pathFile, 'r') as file: 
            self.num_games = file.attrs["numberOfGames"]
            self.num_moves = file.attrs["numberOfMoves"]

    def __len__(self):
        return self.num_moves
    
    def __getitem__(self, idx):
        with h5py.File(self.pathFile, 'r') as file:
            game_group = file['games'][f"game-{idx+1}"]
            move = self.current_serie_moves[self.current_data_position[1]]
            moves_played = self.current_serie_moves[0:self.current_data_position[1]]
            scoreForWeight = game_group['averageElo'][()]

            #legal_moves
            legal_moves = self.findLegalMove(moves_played)
            #static_postions
            static_positions = self.get_states(moves_played)
            
        seq = self.encode_move(moves_played)
        self.current_data_position[1] += 1
        return seq, static_positions, legal_moves, scoreForWeight, target
    
class CHESSBot(nn.Module):
    def __init__(self): 
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=24, num_layers=2, batch_first=True)

        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 8 * 8 + 24, 3072)
        self.fc2 = nn.Linear(3072, 4116)

    def forward(self, seq, board_state, seq_length=None, temperature:float=1.0):
        """
        seq: [batch_size, seq_len, 4]
        board_state: [batch_size, 8, 8, 8]
        temperature: Controls randomness of the output selcted by model
        """
        cnn_out = nn.ReLU()(self.conv1(board_state))
        cnn_out = nn.ReLU()(self.conv2(cnn_out))
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        if seq_length is not None:
            seq_packed = pack_padded_sequence(seq, seq_length.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(seq_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(seq)

        lstm_out = lstm_out[:, -1, :]
        
        x = torch.cat([cnn_out, lstm_out], dim=1)
        #Try F.relu maybe faster
        x = F.relu()(self.fc1(x))
        x = self.fc2(x)
        
        if temperature != 1.0:
            x = x / temperature

        return x
    
### CHECKPOINT FUNCTION add into training loop
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved at {path} /n Epoch: {epoch} /n Loss: {loss}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {path} /n Epoch: {epoch} /n Loss: {loss}")
    return model, optimizer


#DATASET
pathToDataset = 'Dataset\original\lichess_db_standard_rated_2013-01.h5'

dataset = ChessDataset(pathToDataset)

dataloader = DataLoader(dataset, batch_size=512, shuffle=True) #Rework for batch size, bc every game

#SETTINGS
model = CHESSBot().to(device)
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #Update step_size(epochs num) and gamma
# MAYBE TRY THIS scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)


#TEST
input_size = 10   
hidden_size = 20  
output_size = 4 # Chess Notation   
data = None
target = None 
epochs = 100

def get_weights():
    with h5py.File('Dataset\original\lichess_db_standard_rated_2013-01.h5', 'r') as file:
        return file['sampleWeightGroup']['sampleWeight']
weights = get_weights()

for epoch in range(epochs):  
    for batch in dataloader:
        seq, static_positions, legal_moves, scoreForWeight,  target = batch
        seq, static_positions, target = seq.to(device), static_positions.to(device), target.to(device)

        predictions = model(seq, static_positions).to(device)

        loss = criterion(predictions, target).to(device)
        weighted_loss = (loss * weights[scoreForWeight]).mean()

        weighted_loss.backward()  
        optimizer.step()   

        optimizer.zero_grad()    

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")