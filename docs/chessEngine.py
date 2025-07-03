import numpy as np
import chess

class ChessEngine(): 
    """
    A Chess eingine which is used to simulate a chess game.
    """
    def __init__(self):
        """
        Initializes the chess eingine with an initial board state.

        The board is represented as a hot-encoded 3D numpy array with shape of (8, 8, 8),
        where each row represents a different figure parameter on a specific positon of the board.
        
        Each row of the third dimension represents diffrent parameter of a chess piece:
            0: Pawn
            1: Rook
            2: Knight
            3: Bishop
            4: Queen
            5: King
            6: Black color
            7: White color

        Attributes:
            initial_board (np.ndarray): The initial board state as a 3D numpy array.
            board (np.ndarray): The current board state, which is a copy of the initial board.
            promotions (dict): A dictionary which is used to map promotion characters to their indeces in the board.
            positions (dict): A dictionary which is used to map chess board positions to their indeces in the board.
        """
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
        self.en_passant_possible ={
            "location": []
        }
    def board_reset(self):
        """
        Resets the board to it's initial state.

        This method copies the initial board state to the current board.
        """
        self.board = self.initial_board.copy()
        
    def move(self, value: str): ########## ADD EN PASSANT
        """
        Moves a piece on the board based on the given value.

        Args:
            value (str): A string representing the move in chess notation, 
                         which should be written in the Universal Chess Interface Notation.
                         For example, 4 characters:
                                                    "d2d4", "g8f6", "c2c4"
                                      5 characters:
                                                    "d2d4q", "g8f6n", "c2c4b", where the last character represents the promotion piece.
        """
        # Handles special moves, which represent castling moves.
        castling_moves = [ "e1g1","e1c1","e8g8","e8c8"]
        if value in castling_moves:
            if np.array_equal(self.board[7][4], [0, 0, 0, 0, 1, 0, 0, 1]):
                # White king castling
                if value == "e1g1":
                    self.board[7][6] = self.board[7][4]
                    self.board[7][4] = [0, 0, 0, 0, 0, 0, 0, 0]
                    self.board[7][5] = self.board[7][7]
                    self.board[7][7] = [0, 0, 0, 0, 0, 0, 0, 0]
                elif value == "e1c1":
                    self.board[7][2] = self.board[7][4]
                    self.board[7][4] = [0, 0, 0, 0, 0, 0, 0, 0]
                    self.board[7][3] = self.board[7][0]
                    self.board[7][0] = [0, 0, 0, 0, 0, 0, 0, 0]
                return
            if np.array_equal(self.board[0][4], [0, 0, 0, 0, 1, 0, 1, 0]): 
                # Black king castling
                if value == "e8g8":
                    self.board[0][6] = self.board[0][4]
                    self.board[0][4] = [0, 0, 0, 0, 0, 0, 0, 0]
                    self.board[0][5] = self.board[0][7]
                    self.board[0][7] = [0, 0, 0, 0, 0, 0, 0, 0]
                elif value == "e8c8":
                    self.board[0][2] = self.board[0][4]
                    self.board[0][4] = [0, 0, 0, 0, 0, 0, 0, 0]
                    self.board[0][3] = self.board[0][0]
                    self.board[0][0] = [0, 0, 0, 0, 0, 0, 0, 0]
                return
        # Distributes the value to the SELECT and POSITION lists.
        SELECT = []     #[file, rank]
        POSITION = []   #[file, rank]
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
        
        # Handles promotions
        if promotion:
            self.board[POSITION[1]][POSITION[0]] = [0, 0, 0, 0, 0, 0, self.board[SELECT[1]][SELECT[0]][5], self.board[SELECT[1]][SELECT[0]][6]]
            self.board[POSITION[1]][POSITION[0]][promotion-1] = 1
        else:   
            if self.board[SELECT[1]][SELECT[0]][0] == 1 and len(self.en_passant_possible["location"]) > 0 and SELECT[0] != POSITION[0]:
                # En passant move
                self.board[SELECT[1]][SELECT[0]] = [0, 0, 0, 0, 0, 0, 0, 0]
                self.board[POSITION[1]][POSITION[0]] = [1, 0, 0, 0, 0, 1, 0, 1]  
                self.board[self.en_passant_possible["location"][1]][self.en_passant_possible["location"][0]] = [0, 0, 0, 0, 0, 0, 0, 0]
                self.en_passant_possible["location"] = []
            self.board[POSITION[1]][POSITION[0]] = self.board[SELECT[1]][SELECT[0]]
        
        # En passant history handling
        if self.board[SELECT[1]][SELECT[0]][0] == 1 and (abs(SELECT[1] - POSITION[1]) == 2):
            self.en_passant_possible["location"] = [POSITION[0], POSITION[1]]

        else:
            self.en_passant_possible["location"] = []

        self.board[SELECT[1]][SELECT[0]] = [0, 0, 0, 0, 0, 0, 0, 0]

    def check_for_promotion(self, move):
        """
        Checks and reurns the promotion piece if the move is a promotion.
        
        Args:
            move (list): A list representing the move, where the last element is the promotion piece if it exists.
                         Should be written in the Universal Chess Interface Notation.
                         For example, "d2d4", "g8f6", "c2c4", etc.

        Returns:
            int: 1 if the move is a promotion, otherwise 0.
        """
        return self.promotions[move[4]] if len(move) == 5 else 0
    
    def get_states(self, game):
        """
        Gets all the states of the game as a numpy array.
        Return a numpy array of a board representation in shape of [num_moves, 8, 8, 8].
        """
        arr = []
        history_moves = []
        state_notation = []
        self.board_reset()
        for move in game:
            self.move(move)
            arr.append(np.array(self.board)) #Creates a copy
            state_notation.append(move)
            history_moves.append(state_notation.copy())  
        return np.array(arr), history_moves

    def encode_move(self, move):
        """
        Helper function to encode a single move string into a encoded format.
        """
        return [self.positions[move[0]], int(move[1]), self.positions[move[2]], int(move[3])]
    
    def findLegalMove(self, moves, encode=False):
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
            if encode:
                encoded = [self.encode_move(m.uci()) for m in board.legal_moves]
                legal_moves.append(np.array(encoded))
            else: 
                legal_moves.append([m.uci() for m in board.legal_moves])
        return legal_moves
    
"""
# ADD CONTEXT here for use
"""