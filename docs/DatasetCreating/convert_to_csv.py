import chess.pgn
import numpy as np
import pandas as pd
import re
import os
import sys
from multiprocessing import Pool
import time

def converting_elo(elo):
    """
    Convert ELO rating to integer or None if invalid.
    """
    try:
        return int(elo) if elo and elo.isdigit() else None
    except ValueError:
        return None

def process_game_serialized(game_data):
    """
    Process a serialized chess game to extract relevant data.

    Relevant data includes:
    - Average ELO of both players
    - Whether the white player won (1 for win, 0 for loss)
    - Moves in the game as a single string
    If the game doesn't meet the criteria (e.g., low ELO, abnormal termination),
    it returns None.
    """
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

def pgn_to_dataframe(pgn_file, limit_count=100_000_000):
    """
    Convert a PGN file to a Pandas DataFrame.
    Extremely slow, but works.
    """
    serialized_games = []
    yield_count = 0
    total_games_processed = 0
    total_valid_games = 0
    pool = Pool(processes=16)

    with open(pgn_file, 'r') as f:
        game_count = 1
        while True:
            game = chess.pgn.read_game(f)
            if game_count % 1000 == 0:
                print(f"Reading game {game_count}")
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

            if len(serialized_games) >= 8000:
                chunk_start = time.time()
                results = pool.imap_unordered(process_game_serialized, serialized_games, chunksize=100)
                results = list(results)
                serialized_games = []

                valid_results = [r for r in results if r is not None]
                total_games_processed += len(results)
                total_valid_games += len(valid_results)

                chunk_time = time.time() - chunk_start
                if valid_results:
                    df_chunk = pd.DataFrame(valid_results)
                    yield_count += 1
                    print(f"Yield #{yield_count} | {len(df_chunk)} games | Chunk time: {chunk_time:.2f}s")
                    print(f"→ Total processed: {total_games_processed} | Total valid: {total_valid_games}")

                    # Estimate total time
                    elapsed_time = time.time() - start_time_global
                    est_total_time = (elapsed_time / total_games_processed) * limit_count
                    time_remaining = est_total_time - elapsed_time
                    print(f"⏳ Elapsed: {elapsed_time:.2f}s | Estimated total: {est_total_time/60:.1f} min | Remaining: {time_remaining/60:.1f} min")

                    yield df_chunk

            if game_count >= limit_count:
                print(f"Reached limit of {limit_count} games.")
                break

    if serialized_games:
        chunk_start = time.time()
        results = pool.imap_unordered(process_game_serialized, serialized_games, chunksize=100)
        results = list(results)
        valid_results = [r for r in results if r is not None]
        total_games_processed += len(results)
        total_valid_games += len(valid_results)

        chunk_time = time.time() - chunk_start
        if valid_results:
            df_chunk = pd.DataFrame(valid_results)
            yield_count += 1
            print(f"Yield #{yield_count} | {len(df_chunk)} games | Chunk time: {chunk_time:.2f}s")
            print(f"→ Total processed: {total_games_processed} | Total valid: {total_valid_games}")

            elapsed_time = time.time() - start_time_global
            est_total_time = (elapsed_time / total_games_processed) * limit_count
            time_remaining = est_total_time - elapsed_time
            print(f"⏳ Elapsed: {elapsed_time:.2f}s | Estimated total: {est_total_time/60:.1f} min | Remaining: {time_remaining/60:.1f} min")

            yield df_chunk

    pool.close()
    pool.join()

def main():
    global start_time_global
    start_time_global = time.time()

    file = r"E:\Datasets\SOC\ChessPositions\lichess_db_standard_rated_2025-01-100M.pgn"
    print(f"Processing file: {file}")
    file_name, _ = os.path.splitext(file)

    df_chunks = list(pgn_to_dataframe(file))
    elapsed = time.time() - start_time_global
    print("\n✅ Total processing time:", elapsed, "seconds")

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
