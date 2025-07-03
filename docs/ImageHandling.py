from PIL import Image
import time
import pandas as pd
import numpy as np
import h5py
import glob
import ast

# Sets the numpy print options to display all elements in an array, which is useful for debugging.
np.set_printoptions(threshold=np.inf)



class ImageConverter():
    """
    Class for slicing chess position images into 64 cropped images representing each square on the board.
    """
    def __init__(self, folder_path=None, dataset_path=None):
        """"
        Initializes the ImageConverter with a folder path containing images and a dataset path containing game states.
        """
        if folder_path is not None and dataset_path is not None:
            self.folder_path = folder_path
            self.images_paths = sorted(glob.glob(folder_path + "/*.*"),key=lambda x: int(''.join(filter(str.isdigit, x))))
            self.df = pd.read_csv(dataset_path)
            self.num_samples = len(self.images_paths)

    def __len__(self):
        return len(self.images_paths)
        
    def image_to_array(self, image_path):
        """
        Converts an image file to a numpy array.
        """
        image = Image.open(image_path)
        image.convert('RGB')
        return np.array(image)

    def array_to_image(self, array):
        """
        Converts a numpy array to a PIL Image.
        """
        return Image.fromarray(array)

    def read_states(self, game_state):
        raw = game_state["state"]
        state_list = ast.literal_eval(raw) if isinstance(raw, str) else raw
        state_arr  = np.array(state_list) 

        positions_on_array = {
                                0: "p", 
                                1: "r",
                                2: "n",
                                3: "b", 
                                4: "k", 
                                5: "q"
                            }

        updated_state = []
        step = 0

        for y in range(8):
            for x in range(8):
                square = state_arr[y, x]
                types = [i for i in range(6) if square[i] == 1]
                ids   = [i - 6 for i in range(6, len(square)) if square[i] == 1]

                if len(types) == 1 and len(ids) == 1:
                    piece = positions_on_array[types[0]]
                    pid   = ids[0]
                    updated_state.insert(0, [piece, pid])
                else:
                    updated_state.insert(0, ["e", 2])
                    
                step += 1

        return updated_state

    
    def crop_image(self, image, game_state, image_name = None):
        square = 0
        
        positions = {1 : [66,186],
                     2 : [61, 288],
                     3 : [53, 393],
                     4 : [46, 503],
                     5 : [38, 618],
                     6 : [31, 733],
                     7 : [23, 856],
                     8 : [15, 981]
                     }
        CONST = 115
        array = []
        for Y in range(8):
            for X in range(8):
                x1 = positions[X + 1][0] + (X * CONST)
                y1 = positions[Y + 1][1] 
                x2 = x1 + CONST
                y_start = y1 - CONST
                y_end = y1
                
                cropped_array = image[y_start:y_end, x1:x2]

                array.append(cropped_array)
                figure, color = game_state[square][0], game_state[square][1] 
                img = Image.fromarray(cropped_array)
                if image_name is None:
                    image_name = "state"
                img.save(fr"E:\Datasets\SOC\ChessPositionsRenders\cropped\{image_name}_{Y}_{X}_{figure}_{color}.png")
                square += 1
        return (array)

    def show_array(self, array):
        self.array_to_image(array).show()

    def save_array(self, array, path):
        self.array_to_image(array).save(path)
    
    def generate(self):
        array = []
        for i,image_path in enumerate(self.images_paths):
            print(f"Processing image {i} out of {image_path}")
            game_state = self.df.iloc[i]
            image = self.image_to_array(image_path)
            array.append(self.crop_image(image, self.read_states(game_state)), "state_" + str(i))
            print(f"Image {i} converted to array out of {len(self.images_paths)}")
            
        print(f"Total number of images: {len(array)}")
           
            


converter = ImageConverter(r"E:\Datasets\SOC\ChessPositionsRenders\state", r"E:\Datasets\SOC\ChessPositionsRenders\dataset_info.csv")

converter.generate()

DATASET_PATH = r"E:\Datasets\SOC\ChessPositionsRenders\pixels_values.h5"
CHUNK_SIZE = 1000
### COMPLETED ###
class ImageSampler():
    def __init__(self, dataset_path, chunk_size):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.map = {}
        self.dataset_name = 'images'  #Change this to the name of the dataset in the h5 file
        self.secondary_dataset_name = 'labels'  #Change this to the name of the dataset in the h5 file
    
    def __len__(self):
        with h5py.File(self.dataset_path, 'r') as file:
            return len(file[self.dataset_name]) 
    
    def load_chunk(self, chunk_id):
        with h5py.File(self.dataset_path, 'r') as file:
            start = chunk_id * self.chunk_size
            end = min(start + self.chunk_size, len(file[self.dataset_name]))
            images = file[self.dataset_name][start:end]
            labels = file[self.secondary_dataset_name][start:end]
            self.update_map(chunk_id, [images,labels])
            return [images, labels]
    
    def update_map(self, chunk_id, images, labels):
        start_idx = chunk_id * self.chunk_size
        for idx, (image, label) in enumerate(zip(images, labels)):
            self.map[start_idx + idx] = (image, label)
        
    def __getitem__(self, idx):
        if idx in self.map:
            return self.map[idx]
        
        chunk_id = idx // self.chunk_size
        self.load_chunk(chunk_id)
        return self.map[idx]

class ImageStorer():
    def __init__(self, dataset_path, chunk_size, images_folder_path):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.dataset_name = 'images'
        self.secondary_dataset_name = 'labels'
        self.map = self.dataset_grabber(images_folder_path)
        
    def dataset_grabber(self, dataset_file_location):
        return glob.glob(dataset_file_location + "/*.*")  

    def store(self, data):
        with h5py.File(self.dataset_path, 'w') as file:
            file.attrs['name'] = 'Chess Images Dataset'
            file.attrs['totalNumImages'] = len(data)
            file.create_dataset(self.dataset_name, data=[image for image, _ in data])
            file.create_dataset(self.secondary_dataset_name, data=[label for _, label in data])
    