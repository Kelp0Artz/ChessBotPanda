import torch 
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
        super().__init__()
        self.pathFile = pathFile
        self.games = h5py.File(self.pathFile, 'r')['games']
        self.weights = h5py.File(self.pathFile, 'r')['sampleWeightGroup']['sampleWeight']
        with h5py.File(self.pathFile, 'r') as file:
            self.num_games = file.attrs["numberOfGames"]
        self.game_
    
    def __len__(self):
        return self.num_games
    
    def __getitem__(self, idx):
        with h5py.File(self.pathFile, 'r') as file:
            game_group = file['games'][f"game-{idx+1}"]
            moves = game_group['moves'][()]
            moves = moves.decode('utf-8').split()
            #legal_moves
            legal_moves = self.findLegalMove(moves)
            #static_postions
            static_positions = self.get_states(moves)
        seq = self.encode_move(moves)
        target = []
        return [seq, target],[static_positions], legal_moves
    
class CHESSBot(nn.Module):
    def __init__(self): 
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=24, num_layers=2, batch_first=True)

        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 8 * 8 + 24, 128)
        self.fc2 = nn.Linear(128, 8*8*8)
        self.fc3 = nn.Linear(8*8*8, 4116)

    def forward(self, seq, board_state):

        cnn_out = nn.ReLU()(self.conv1(board_state))
        cnn_out = nn.ReLU()(self.conv2(cnn_out))
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        lstm_out, _ = self.lstm(seq)
        lstm_out = lstm_out[:, -1, :]
        
        x = torch.cat([cnn_out, lstm_out], dim=1)

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        
        return x




#SETTINGS
model = CHESSBot().to(device)
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    predictions = model(data)
    loss = criterion(predictions, target).to(device)
    weighted_loss = (loss * weights[weight]).mean()
    loss.backward()  
    optimizer.step()   
    optimizer.zero_grad()    

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")