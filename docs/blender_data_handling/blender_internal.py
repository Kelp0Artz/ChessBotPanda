import bpy
import sys
import subprocess
import numpy as np
import csv
import random
import math
import h5py
import chess.pgn
import os


# THIS CLASS WAS USED FROM CHESSBOT PROJECT

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
            if np.array_equal(self.board[7][4], [0, 0, 0, 0, 1, 0, 0, 1]): ###FIX either one of them because the color shouldn't be same
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
            if np.array_equal(self.board[0][4], [0, 0, 0, 0, 1, 0, 0, 1]): ###FIX either one of them because the color shouldn't be same
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
            self.board[POSITION[1]][POSITION[0]] = self.board[SELECT[1]][SELECT[0]]
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

class Sampler(ChessEngine):
    """
    A Sampler that precomputes all board states for each game once, then 
    serves them via __getitem__ without repeated resets.
    """
    def __init__(self, dataset_path, size, from_index=0):
        super().__init__()
        self.file_path = dataset_path
        self.map = {}
        self.game_states = {}
        self.offsets = []  # list of tuples (key, offset)
        self.total_samples = 0
        self.from_index = from_index

        # Load raw data from the dataset file
        try:
            with h5py.File(self.file_path, 'r') as f:
                raw_games = f['games/moves'][from_index:from_index+size+1]
        except Exception as e:
            print(f"Error opening dataset: {e}")
            raw_games = []

        # Decodes and --- decode & index, then precompute states ---
        cumulative = 0
        key = 0
        for raw in raw_games:
            game = raw.decode('utf-8').split()
            if not game:
                continue

            # Save the raw move list
            self.map[key] = game
            

            # Precompute the full sequence of states for this game
            states, history_moves = super().get_states(game)  # resets board once and builds [N,8,8,8]
            
            self.game_states[key] = [states, history_moves]

            # Record offset for indexing
            self.offsets.append((key, cumulative))
            cumulative += len(states)
            key += 1

        self.total_samples = cumulative

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx, game_idx=None):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range (0..{self.total_samples-1})")

        game_id = 0
        for key, offset in self.offsets:

            states, history_moves = self.game_states[key]
            if idx < offset + len(states):
                local_idx = idx - offset
                return states[local_idx], history_moves[local_idx], game_id, states[local_idx-1] if local_idx > 0 else self.initial_board
            game_id += 1
        
        raise IndexError(f"Index {idx} could not be mapped to a game.")


        
            
class RenderCreation:
    """
    Class used for rendering chess positions in a simulated enviroment using Blender.
    """
    def __init__(self, sampler, save_path_folder, start_x, start_y, start_idx = 0, square_size = 50):
        """
        Initializes the RenderCreation class.
        
        Args:
            sampler (Sampler): An instance of the Sampler class to provide chess positions.
            save_path_folder (str): Represents the path to the folder where the rendered images will be saved.
            start_x (int): The starting x cordinate for placing the chess piece on the board.
            start_y (int): The starting y cordinate for placing the chess piece on the board.
            start_idx (int): ADD LATER IDK WHAT DOES IT DO 
            square_size (int): The difference of a value with which should another figure move. ###!!!!!!!!!!!

        """
        super().__init__()
        self.states = {}
        self.copied_pieces = []
        self.sampler = sampler
        self.positions_on_array = {
                                    0:"Pawn",
                                    1:"Rook",
                                    2:"Knight",
                                    3:"Bishop",
                                    5:"Queen",
                                    4:"KING"
                                    }
                                    
        self.board_position = [{"x": start_x + col * square_size, "y": start_y + row * square_size} for row in range(8) for col in range(8)]
        self.pieces_info = None #Stores everything important BRO
        self.save_path_folder = save_path_folder
        self.HDRI_folder_path = r"E:\Datasets\SOC\HDRI\ALL"
        self.HDRI_collection = [f for f in os.listdir(self.HDRI_folder_path)]
        

        self.state_seen_set = set()
        self.start_idx = start_idx
    
    def copies_checker(self, state):
        """
        Checks if the state has been seen before.
        This method uses a set to store hashes of the states to quickly check for duplicates.

        Args:
            state (list): A list representing the chess position, where each element is a notation of a chess moves,
                          which should be written in the Universal Chess Interface Notation.
        
        Returns:
            bool: True if the state has been seen before, otherweise False.
        """
        if len(state) >2:
            return False
        game_hash = hash(tuple(state))  
        if game_hash in self.state_seen_set:
            return True
        else:
            self.state_seen_set.add(game_hash)
            return False
       
    def set_hdri(self, hdri_path):
        """ 
        Sets the HDRI environment texture for the world in Blender.
        HDRI or High Dynamic Range Imaging is a type of an image that contains information about the light in a scene,
        which can be used to create realistic lighting and reflections in 3D rendering.

        Args:
            hdri_path (str): The file path to the HDRI image.
        """
        world = bpy.data.worlds["World.002"]
        node_tree = world.node_tree
        env_node = node_tree.nodes.get("Environment Texture")
    
        if not env_node:
            env_node = node_tree.nodes.new('ShaderNodeTexEnvironment')
            background = node_tree.nodes['Background']
            node_tree.links.new(env_node.outputs['Color'], background.inputs['Color'])

        env_node.image = bpy.data.images.load(hdri_path)
        
    def clear(self, collection_name="Duplicates"):
        """
        Clears rendering zone and deletes unused mesh data.

        Args:
            collection_name (str): The name of the collection to clear. Default is "Duplicates", which is used to store copied chess pieces.
        """
        collection = bpy.data.collections.get(collection_name)

        if collection:
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        
            # Clean up unused mesh data
            for mesh in bpy.data.meshes:
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)

        else:
            print(f"Collection '{collection_name}' not found!")
            
    def copy_position_mesh(self, obj_name, position, color_state, collection_name = "Duplicates"):
        """
        Copies a mesh object to a specified position and color state, and links it to a collection.

        Args:
        
        """
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]  

        
            obj_copy = obj.copy()
            obj_copy.data = obj.data.copy()  

            bpy.context.collection.objects.link(obj_copy)

            obj_copy.location = position
            
            collection = bpy.data.collections.get(collection_name)
            if obj_copy and collection:
                for coll in obj_copy.users_collection:
                    coll.objects.unlink(obj_copy)
                collection.objects.link(obj_copy)
       
            obj_copy.name = f"copy_{len(self.copied_pieces) + 1}"
            
            self.copied_pieces.append(obj_copy)  

            if "WhitePieceMaterial" and "BlackPiecesMaterial" in bpy.data.materials:
                if color_state:
                    material = bpy.data.materials["WhitePieceMaterial"]
                    random_angle = random.uniform(135,225)
                    obj_copy.rotation_euler.rotate_axis("Z", math.radians(random_angle))

                else: 
                    material = bpy.data.materials["BlackPiecesMaterial"]

                if len(obj_copy.data.materials) > 0:
                    obj_copy.data.materials[0] = material
                else:
                    obj_copy.data.materials.append(material)
            else:
                print("Material 'WhitePieceMaterial' not found.")
        
            return obj_copy
        else:
            print(f"Object '{obj_name}' not found in the scene.")  
            return None
        
    def read_states(self, state, return_value = None): #BIG ERRORS FIX NOW!!!!!!!!!!! UNTIL NOT DELETED
        """
        Updates a self.pieces_info about the position of the pieces.
        """
        updated_state = {}
        current_state = []
        step = 0

        for y, columns in enumerate(state):
            for x, column in enumerate(columns):
                for idx, encoded_pos in enumerate(column):  
                    
                    if encoded_pos == 1 and idx <= 5:      
                        current_state = [self.positions_on_array[idx], step]
                    elif current_state != [] and encoded_pos == 1:
                        updated_state[current_state[1]] = [current_state[0], idx - 6]  #ID = current_state[1] ;D

                step += 1 

        self.pieces_info = updated_state
        if return_value != None:
            return updated_state
    
    def place_figures(self):
        """
        Places meshes of figures on a chessboard, which are slightly dispositioned
        for better data generalization.

        Variables:
            DIFF (int): The maximum random offset applied to the position of each piece.
        """
        DIFF = 10
        for key in self.pieces_info.keys():
            # Calculate row and column from key.
            row = key // 8
            col = key % 8
            # Mirror the column index.
            new_index = row * 8 + (7 - col)
        
            # Get the position from the mirrored index.
            base_pos = self.board_position[new_index]
        
            # Apply random offset.
            X_diff = random.randint(-DIFF, DIFF)
            Y_diff = random.randint(-DIFF, DIFF)
            X = base_pos["x"] + X_diff
            Y = base_pos["y"] + Y_diff
        
            position = (X, Y, 0)
            # self.pieces_info[key] is expected to be in the format [object_name, color_state]
            self.copy_position_mesh(self.pieces_info[key][0], position, self.pieces_info[key][1])

   
               
    def add_row_to_csv(self, row, image_path, previous_game_state):
        """
        Adds a row in selected dataset adress for better data tracking. 
        """
        row_str = np.array2string(row, separator=',', max_line_width=np.inf)
        row_str = row_str.replace('\n', ' ')

        previous_game_state_str = np.array2string(previous_game_state, separator=',', max_line_width=np.inf)
        previous_game_state_str = previous_game_state_str.replace('\n', ' ')

        with open(self.save_path_folder + "\\dataset_info.csv", mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([image_path, row_str, previous_game_state_str])

    def create_renders(self):
        """
        Renders a scene.
        """
        total_num_samples = len(self.sampler)
        id  =  self.start_idx
        
        for idx in range(total_num_samples):
            (state, chess_notation, game_id, previous_game_state) = self.sampler[idx]
            
            if self.copies_checker(chess_notation):
                print("State have been already seen")
                continue

            self.read_states(state, True) #ADD
            self.place_figures()
            
            
            hdri_name = random.choice(self.HDRI_collection)
            path =  self.save_path_folder +f"\\state\\" + f"\\state-{id}.png"
            self.add_row_to_csv(state, path, previous_game_state)
            bpy.context.scene.render.filepath = path
            
            hdri_path = os.path.join(self.HDRI_folder_path, hdri_name)
            self.set_hdri(hdri_path)
            
            bpy.ops.render.render(write_still=True)
            self.clear()
            id += 1

    def find_state(self, state_idx):
        """
        Finds a state in the dataset.
        """
        for state_num in range(state_idx + 1):
            (state, chess_notation, game_id, previous_game_state) = self.sampler[state_num]

            if self.copies_checker(chess_notation):
                print("State have been already seen")
                continue

            if state_num == state_idx:
                print(f"State is in the {game_id}th game")

os.system('cls')   
sampler = Sampler(r"E:\Datasets\SOC\ChessPositions\lichess_db_2021-03.h5", 1000, 21)
chess_engine = ChessEngine()
render_file_info = "E:\\Datasets\\SOC\\ChessPositionsRenders"


render = RenderCreation(sampler, render_file_info, start_x = 0, start_y = 0, start_idx = 1384)
render.create_renders()

# To run this script in Blender, is recommended to use the following command in the terminal, 
# which should help with GUI freezing issues:
#   blender --background --python main.py




