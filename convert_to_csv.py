import chess.pgn
import pandas as pd
import re
import os
from multiprocessing import Pool

def converting_elo(elo):
    try:
        return int(elo) if elo and elo.isdigit() else None
    except ValueError:
        return None

def process_game(game):
    termination = game.headers.get('Termination', '')
    whiteWon = game.headers.get('Result', '')
    whiteElo = converting_elo(game.headers.get('WhiteElo', ''))
    blackElo = converting_elo(game.headers.get('BlackElo', ''))
    if whiteElo is None or blackElo is None:
        return None
    averageElo = (whiteElo + blackElo) // 2
    
    if whiteWon == "0-0" or termination != "Normal" or averageElo < 750:
        return None

    game_data = {
        "AverageElo" : averageElo,
        "WhiteWon": (1 if whiteWon == '1-0' else 0),
        "Moves": ' '.join([move.uci() for move in game.mainline()])
    }

    return game_data

def pgn_to_dataframe(pgn_file, limit_count=20_000_000):
    games = []
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

            games.append(game)

            if len(games) >= 4000:  
                results = pool.map(process_game, games)  
                games = []
                results = [result for result in results if result is not None]
                if results:
                    df_chunk = pd.DataFrame(results)  
                    yield_count += 1  
                    print(f"Yield #{yield_count} with {len(df_chunk)} games")  
                    yield df_chunk  
            
            if game_count >= limit_count: 
                print(f"Processed {game_count} games, stopping.")
                break

    if games:
        results = pool.map(process_game, games)
        results = [result for result in results if result is not None]
        
        if results:
            df_chunk = pd.DataFrame(results)
            yield_count += 1  
            print(f"Yield #{yield_count} with {len(df_chunk)} games")  
            yield df_chunk

    pool.close()
    pool.join()

def main():
    files = ['Dataset/original/lichess_db_standard_rated_2024-12.pgn']
    for file in files:
        print(f"Processing file: {file}") 
        file_name, _ = os.path.splitext(file)
        df_chunks = pgn_to_dataframe(file)
        full_df = pd.concat(df_chunks, ignore_index=True)
        print(f"Saving to CSV: {file_name}_result_0_1.csv")
        full_df.to_csv(f'{file_name}_result_0_1.csv', index=False)
        
if __name__ == '__main__':
    main()
