import numpy as np
import pandas as pd
import csv
import h5py
import math

class ChessEngine():
    def convert_to_h5py(self, inputFilePath, outputFilePath):
        """adresses = []
        countGame = 0
        try:
            with open(inputFilePath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    adresses.append(line.strip())
        except FileNotFoundError:
            print('File not found')
            return -1"""
        adresses = [inputFilePath]
        
        array = []
        for i in range(1, 3201):
            normalized_value = (i - 1) / (3200 - 1)  
            sigmoid_value = 1 / (1 + math.exp(-12 * (normalized_value - 0.5)))  
            array.append(np.float32(sigmoid_value))
        
        with h5py.File(outputFilePath, 'w') as file:
            countGame = 1
            countMoves = 0
            file.attrs['name'] = 'Chess Dataset'
            Additional = file.create_group('sampleWeightGroup')
            Additional.create_dataset('sampleWeight', data=array)
            group = file.create_group('games')
            for adress in adresses:
                with open(adress, 'r', newline='') as fileCSV:
                    reader = csv.reader(fileCSV)
                    headers = next(reader)
                    for row in reader:
                        game_group = group.create_group(f'game-{countGame}')
                        averageElo = row[0]
                        whiteWon = row[1]
                        moves = row[2]
                        game_group.attrs['averageElo'] = averageElo    
                        game_group.attrs['whiteWon'] = whiteWon
                        game_group.create_dataset('moves', data=moves)
                        print(f'Game {countGame} has been added')
                        
                        countMoves += len(moves.split())
                        countGame += 1

            file.attrs['numberOfGames'] = countGame
            file.attrs['numberOfMoves'] = countMoves

Engine = ChessEngine()
Engine.convert_to_h5py('Dataset/lichess_db_standard_rated_2013-01_result_0_1.csv', 'Dataset/lichess_db_standard_rated_2013-01.h5')


