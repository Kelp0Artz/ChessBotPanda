import chess.pgn
import numpy as np
import pandas as pd
import re
import os
import sys
from multiprocessing import Pool
import time

def converting_elo(elo):
    try:
        return int(elo) if elo and elo.isdigit() else None
    except ValueError:
        return None

def process_game_serialized(game_data):
    try:
        headers = game_data['headers']
        moves = game_data['moves']
        termination = headers.get('Termination', '')
        whiteWon = headers.get('Result', '')
        whiteElo = converting_elo(headers.get('WhiteElo', ''))
        blackElo = converting_elo(headers.get('BlackElo', ''))
        if whiteElo is None or blackElo is None:
            return None
        averageElo = (whiteElo + blackElo) // 2

        if whiteWon == "0-0" or termination != "Normal" or averageElo < 750:
            return None

        return {
            "AverageElo": averageElo,
            "WhiteWon": 1 if whiteWon == '1-0' else 0,
            "Moves": ' '.join(moves)
        }
    except Exception as e:
        print("Error processing game:", e)
        return None

def pgn_to_dataframe(pgn_file, limit_count=5_000_000):
    serialized_games = []
    yield_count = 0
    pool = Pool(processes=16)

    with open(pgn_file, 'r') as f:
        game_count = 1
        while True:
            game = chess.pgn.read_game(f)
            if game_count % 1000 == 0:
                print(f"Processing game {game_count}")
            game_count += 1
            if game is None:
                break

            try:
                headers = dict(game.headers)
                moves = [move.uci() for move in game.mainline()]
                serialized_games.append({
                    'headers': headers,
                    'moves': moves
                })
            except Exception as e:
                print(f"Error serializing game {game_count}: {e}")
                continue

            # Increase chunk size to reduce overhead.
            if len(serialized_games) >= 8000:
                # Using imap_unordered with a specified chunksize
                results = pool.imap_unordered(process_game_serialized, serialized_games, chunksize=100)
                results = list(results)
                serialized_games = []
                results = [result for result in results if result is not None]
                if results:
                    df_chunk = pd.DataFrame(results)
                    yield_count += 1
                    print(f"Yield #{yield_count} with {len(df_chunk)} games")
                    yield df_chunk

            if game_count >= limit_count:
                print(f"Processed {game_count} games, stopping.")
                break

    if serialized_games:
        results = pool.imap_unordered(process_game_serialized, serialized_games, chunksize=100)
        results = list(results)
        results = [result for result in results if result is not None]
        if results:
            df_chunk = pd.DataFrame(results)
            yield_count += 1
            print(f"Yield #{yield_count} with {len(df_chunk)} games")
            yield df_chunk

    pool.close()
    pool.join()

def main():
    file = r'C:\Users\juraj\Documents\GitHub\ChessBotPanda\datasets\lichess_db_standard_rated_2013-01 (1).pgn'
    print(f"Processing file: {file}")
    file_name, _ = os.path.splitext(file)
    
    start_time = time.time()
    df_chunks = list(pgn_to_dataframe(file))
    elapsed = time.time() - start_time
    print("Total processing time:", elapsed, "seconds")
    
    if df_chunks:
        print("Chunk sizes:", [len(chunk) for chunk in df_chunks])
        full_df = pd.concat(df_chunks, ignore_index=True)
    else:
        print("No valid game data processed.")
        full_df = pd.DataFrame()
        
    output_csv = f'{file_name}_result_0_1.csv'
    print(f"Saving to CSV: {output_csv}")
    full_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
