from PIL import Image
import time
import pandas as pd
import numpy as np
import h5py
import glob
np.set_printoptions(threshold=np.inf)



class ImageConverter():
    def __init__(self, folder_path, dataset_path):
        self.folder_path = folder_path
        self.images_paths = glob.glob(folder_path + "/*.*")  # Get all files
        self.df = pd.read_csv(dataset_path)

    def __len__(self):
        return len(self.images_paths)
    
    def image_to_array(self, image_path):
        image = Image.open(image_path)
        image.convert('RGB')
        return np.array(image)

    def array_to_image(self, array):

        return Image.fromarray(array)

    def crop_image(self, image, image_name):
        array = []
        for i in range(8):
            for j in range(8):
                x1 = i * 125
                y1 = j * 125
                x2 = x1 + 125
                y2 = y1 + 125


                cropped_array = image[x1:x2, y1:y2]

                array.append(cropped_array)
               
                img = Image.fromarray(cropped_array)
                img.save(fr"E:\Datasets\SOC\ChessPositionsRenders\cropped\{image_name}_{i}_{j}.png")
        return (array)

    def show_image(self, image):
        image.show()

    def save_image(self, image, path):
        image.save(path)

    def show_array(self, array):
        self.array_to_image(array).show()

    def save_array(self, array, path):
        self.array_to_image(array).save(path)

    def generate(self):
        array = []
        for i,image_path in enumerate(self.images_paths):
            image = self.image_to_array(image_path)
            array.append(self.crop_image(image, "state_" + str(i)))
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
    