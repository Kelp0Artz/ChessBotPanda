import numpy as np
import pandas as pd
import csv

class ChessEngine():
    """
    Which one is A or B?
    <<From Right to Left<<
    [PIECE, COLOR, WHICH ONE]
    """
    def __init__(self):
        self.board = np.zeros((8,8,3))  # 192 Inputs its ok * 8 = 1536 so not so ok
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
        self.type = {
            "A": 1,
            "B": 2,
            "C": 1,
            "D": 2,
            "E": 1,
            "F": 2,
            "G": 1,
            "H": 2
        }
        self.promotions =  {
            "r": 2,
            "n": 3,
            "b" : 4,
            "q" : 5
        }
        self.chess_pieces = {
            (6, 1): "\u2654",
            (6, 2): "\u265A",
            (5, 1): "\u2655",
            (5, 2): "\u265B",
            (4, 1): "\u2657",
            (4, 2): "\u265D",
            (3, 1): "\u2658",
            (3, 2): "\u265E",
            (2, 1): "\u2656",
            (2, 2): "\u265C",
            (1, 1): "\u2659",
            (1, 2): "\u265F",
            (0.0, 0.0): " "
        }
    def get_index_info(self, X, Y):
        return self.board[Y][X]
    
    def create_board(self):
        def add_pawns():
            color = 1
            for player in range(1, 7, 5):
                for piece_pos in range(0, self.board.shape[0]):
                    self.board[player][piece_pos] = [1, color, piece_pos+1]
                color += 1

        def add_rooks():
            color = 1
            for player in range(0, 8, 7):
                num_piece = 0
                for piece_pos in range(0, 8, 7):
                    num_piece +=1
                    self.board[player][piece_pos] = [2, color, num_piece]
                color += 1

        def add_knights():
            color = 1
            for player in range(0, 8, 7):
                num_piece = 0 
                for piece_pos in range(1, 7, 5):
                    num_piece += 1
                    self.board[player][piece_pos] = [3, color, num_piece]
                color += 1

        def add_bishop():
            color = 1
            for player in range(0, 8, 7):
                num_piece = 0  
                for piece_pos in range(2, 6, 3):
                    num_piece += 1
                    self.board[player][piece_pos] = [4, color, num_piece]
                color += 1

        def add_queen_king():
            color = 1
            num_piece = 0  
            for player in range(0, 8, 7):
                self.board[player][3] = [6, color, 1]
                self.board[player][4] = [5, color, 1]

        add_pawns()
        add_rooks()
        add_knights()
        add_bishop()
        add_queen_king()

    def move(self, value):
        SELECT = []
        POSITION = []
        positions = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7
        }
        promotion = False
        for char in value:
            if char in self.promotions and len(POSITION) == 2:
                promotion = self.promotions[char]
                continue
            if char in positions:
                char = int(positions[char])
            else:
                char = int(char)
                char -= 1
                char = -(char)-1
                
            if len(SELECT) < 2:
                SELECT.append(char)
                
            else:
                POSITION.append(char)  

        self.board[POSITION[1]][POSITION[0]] = self.board[SELECT[1]][SELECT[0]] 
        if promotion != False:
            self.board[POSITION[1]][POSITION[0]][0] = promotion
        self.board[SELECT[1]][SELECT[0]] = [0.0,0.0,0.0]  
        
    def show(self):
        show_board = np.empty((8,8), dtype='U4')
        for row_i, row in enumerate(self.board):
            for column_i, column in enumerate(row):
                parameters = (column[0], column[1])
                show_board[row_i][column_i] = self.chess_pieces[parameters]
        
        for row_i, row in enumerate(show_board):
            print(f"{8 - row_i} ", end="")
            print(' '.join(row))
        print("  a b c d e f g h")
    
    def converter(self, file_path, feature_column, output_path):
        # Open the input CSV file for reading
        with open(file_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
            reader = pd.read_csv(infile, chunksize=1)  # Read one row at a time
            writer = csv.writer(outfile)
            
            # Write the header to the output CSV
            writer.writerow(['ID', feature_column])
            
            for chunk in reader:
                row = chunk.iloc[0]  # Get the first (and only) row from the chunk
                self.create_board()
                moves = row[feature_column]
                features = list(moves.split(' '))
                updated_features = []

                for feature in features:
                    self.move(feature)
                    board = self.board.tolist()
                    updated_features.append(board)
                    
                # Convert the updated_features to a string representation
                updated_features_str = str(updated_features)
                
                # Write the processed row to the output CSV
                writer.writerow([row.name, updated_features_str])

# Usage
Engine = ChessEngine()
Engine.converter('Dataset\converted\convert.csv', 'Moves', 'Dataset/converted/convert.csv')
