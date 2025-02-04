import numpy as np
import pandas as pd
import csv
import h5py
import math

class ChessEngine():
    def convert_to_h5py(self, inputFilePath, outputFilePath):
        adresses = [inputFilePath]
        
        array = []
        for i in range(1, 3201):
            normalized_value = (i - 1) / (3200 - 1)  
            sigmoid_value = 1 / (1 + math.exp(-12 * (normalized_value - 0.5)))  
            array.append(np.float32(sigmoid_value))
        
        with h5py.File(outputFilePath, 'w') as file:
            countGame = 0
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
                        countGame += 1
                        game_group = group.create_group(f'game-{countGame}')
                        averageElo = row[0]
                        whiteWon = row[1]
                        moves = row[2].strip()
                        move_list = moves.split()
                        len_moves = len(move_list)
                        game_group.attrs['averageElo'] = averageElo    
                        game_group.attrs['whiteWon'] = whiteWon
                        game_group.attrs['numberOfMoves'] = len_moves
                        game_group.create_dataset('moves', data=move_list)
                        #print(f'Game {countGame} has been added')
                        if countGame % 10000 == 0:
                            print(f'Game {countGame} has been added')
                        countMoves += len_moves
                        

            file.attrs['numberOfGames'] = countGame
            file.attrs['numberOfMoves'] = countMoves

Engine = ChessEngine()
Engine.convert_to_h5py('Dataset/lichess_db_standard_rated_2017-01_result_0_1.csv', 'Dataset/lichess_db_standard_rated_2017-01.h5')


