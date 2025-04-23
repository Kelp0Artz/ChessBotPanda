import numpy as np
import csv
import h5py
import math

# --------------------------#
# DatasetH5Creater          #
# --------------------------#

class DatasetH5Creater():
    """
    This class is used to convert a CSV file containing chess game data into an HDF5 file.
    Dataset contains the following attributes:
    - name: Name of the dataset
    - totalNumGames: Total number of games in the dataset
    - totalNumSamples: Total number of samples in the dataset
    - sampleWeight: Weight of EACH OBTAINABLE VALUE in the dataset
    - averageElo: Average Elo rating of the players in the dataset
    - whiteWon: Result of the game (1 if white won, 0 if black won)
    """
    def total_num_samples(self):
        return 
    def convert_to_h5py(self, inputFilePath, outputFilePath):
        adresses = [inputFilePath]
        sample_weights = []
        for i in range(1, 3201):
            normalized_value = (i - 1) / (3200 - 1)  
            sigmoid_value = 1 / (1 + math.exp(-12 * (normalized_value - 0.5)))  
            sample_weights.append(np.float32(sigmoid_value))
        games_data = []  
        elos = []
        results = []
        total_num_moves = 0
        for adress in adresses:
            with open(adress, 'r', newline='') as fileCSV:
                reader = csv.reader(fileCSV)
                next(reader)      
                for row in reader:
                    elos.append(float(row[0]))  
                    results.append(int(row[1]))  
                    moves = row[2].split() # Check if it does interfere with moves_dataset[:] = [" ".join(moves) for moves in games_data] 
                    games_data.append(moves)  
                    total_num_moves += len(moves)

        elos = np.array(elos, dtype=np.float32)
        results = np.array(results, dtype=np.int8) 
        with h5py.File(outputFilePath, 'w') as file:
            file.attrs['name'] = 'Chess Dataset'  
            total_num_games = len(games_data)
            file.attrs['totalNumGames'] = total_num_games
            file.attrs['totalNumSamples'] = total_num_moves - total_num_games
            file.create_group('sampleWeightGroup').create_dataset('sampleWeight', data=sample_weights)
            games_group = file.create_group('games')
            games_group.create_dataset('averageElo', data=elos)
            games_group.create_dataset('whiteWon', data=results)
            dt = h5py.string_dtype(encoding="utf-8")  
            moves_dataset = games_group.create_dataset('moves', (len(games_data),), dtype=dt)
            moves_dataset[:] = [" ".join(moves) for moves in games_data]  

        print(f'Saved {len(games_data)} games successfully!')
converter = DatasetH5Creater()
converter.convert_to_h5py(r"C:\Users\juraj\Documents\GitHub\ChessBotPanda\datasets\lichess_db_standard_rated_2013-01 (1)_result_0_1.csv", r"C:\Users\juraj\Documents\GitHub\ChessBotPanda\datasets\sample.h5")

