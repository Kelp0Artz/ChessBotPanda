import numpy as np
import pandas as pd
class LegalMoves():
    """
    INFO
    ----
    Functions in this class return all legal moves of selected pieces.
    
    RETURNS
    -------
        array : ndarray
    
    """
   


class ChessEngine(LegalMoves):
    """
    Which one is A or B?
    <<From Right to Left<<
    [PIECE, COLOR, WHICH ONE]
    """
    def __init__(self):
        super().__init__
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
    def convert_chess_notation(self, value):
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
        return [SELECT, POSITION]
    
    def get_index_info(self, X, Y):
        return self.board[Y][X]
    
    def play(self):
        while True:
            cordination = input()
            if self.move(cordination) == -1:
                break
            self.show()

    def create_board(self):
        """
        Just creates a playing board maybe later just delete
        """
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
        SELECT , POSITION  = self.convert_chess_notation(value)
        if self.board[SELECT[1]][SELECT[0]][0] == 0:
            print('Wrong Input')
            return -1
        elif self.board[SELECT[1]][SELECT[0]][1] == self.board[POSITION[1]][POSITION[0]][1]:
            print('wtf')
            return -1
            
        else:
            self.board[POSITION[1]][POSITION[0]] = self.board[SELECT[1]][SELECT[0]] 
            """if promotion != False:
                self.board[POSITION[1]][POSITION[0]][0] = promotion
            self.board[SELECT[1]][SELECT[0]] = [0.0,0.0,0.0]  """
    
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
    
    def process_moves(self, moves):
        self.create_board()
        updated_features = []
        features = moves.split(' ')
        for feature in features:
            self.move(feature)
            board = self.board.copy()  # Assuming board can be copied
            updated_features.append(board)
        return updated_features

    def converter(self, file_path, feature_column, chunk_size=1000):
        id = 0
        part = 0
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
        
        for chunk in chunk_iter:
            if feature_column not in chunk.columns:
                raise ValueError(f"Column '{feature_column}' not found in the CSV file.")
            
            chunk[feature_column] = chunk[feature_column].apply(self.process_moves)
            id += len(chunk)

            # Save the processed chunk
            chunk.to_csv(f'Dataset/converted/converted_part_{part}.csv', index=False)
            part += 1

            print(f"Processed {id} rows, saved part {part}")
        
        print("Conversion complete.")
    

        
# Example usage
Engine = ChessEngine()

#Engine.converter('Dataset/converted/test_sample_to_convert.csv', 'Moves', chunk_size=1000)

Engine.create_board()
print(Engine.board.shape)
#folder = ['Dataset\converted\lichess_db_standard_rated_2013-02.csv', 'Dataset\converted\lichess_db_standard_rated_2013-03.csv', ]
#for file in folder:
#---------Engine.converter('Dataset/converted/test_sample_to_convert.csv', 'Moves')
#Engine.converter('Dataset\converted\lichess_db_standard_rated_2013-01.csv', 'Moves')
#Engine.play()
#print(Engine.get_index_info(1,0))
#Engine.convert_chess_notation('d8d7')
print(Engine.board)
"""Engine.move('c1d3')
Engine.show()
Engine.move('c1d2')
Engine.show()"""