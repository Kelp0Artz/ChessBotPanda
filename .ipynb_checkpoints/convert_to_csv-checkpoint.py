import chess.pgn
import pandas as pd
import re
import os

def pgn_to_dataframe(pgn_file):
    games = []
    num_games = 0

    def converting_elo(elo):
        try:
            return int(elo)
        except ValueError:
            return None

    with open(pgn_file, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  
                
            site = game.headers.get('Site', '')
            match = re.search(r'lichess.org/(\w+)', site)
            site_id = match.group(1) if match else site
            
            game_data = {
                #'Event': game.headers.get('Event', ''),
                #'Site': site_id,  
                #'UTCDate': game.headers.get('UTCDate', ''),
                #'UTCTime': game.headers.get('UTCTime', ''),
                #'White': game.headers.get('White', ''),
                'WhiteElo': converting_elo(game.headers.get('WhiteElo', '')),
                #'WhiteRatingDiff': game.headers.get('WhiteRatingDiff', ''),
                #'Black': game.headers.get('Black', ''),
                'BlackElo': converting_elo(game.headers.get('BlackElo', '')),
                #'BlackRatingDiff': game.headers.get('BlackRatingDiff', ''),
                #'Result': game.headers.get('Result', ''),
                'Moves': ' '.join([move.uci() for move in game.mainline()]) 
            }
            
            games.append(game_data)
            num_games += 1
            if num_games % 1000 == 0:
                print(num_games)
    
    df = pd.DataFrame(games)
    return df
list = ['Dataset\original\lichess_db_standard_rated_2013-01.pgn']
for file in list:
    file_name, _ = os.path.splitext(file)
    df = pgn_to_dataframe(file) 
    df.to_csv(f'{file_name}.csv', index=False)
