import chess.pgn
import pandas as pd

# Function to parse a PGN file and convert a subset of games to a DataFrame
def pgn_to_dataframe(pgn_file, num_games=10):
    games = []
    
    # Open the PGN file
    with open(pgn_file, 'r') as f:
        for i in range(num_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break  # End of file
            
            # Extract game metadata
            game_data = {
                'Event': game.headers.get('Event', ''),
                'Site': game.headers.get('Site', ''),
                'Date': game.headers.get('Date', ''),
                'Round': game.headers.get('Round', ''),
                'White': game.headers.get('White', ''),
                'Black': game.headers.get('Black', ''),
                'Result': game.headers.get('Result', ''),
                'Moves': ' '.join([move.uci() for move in game.mainline()])  # UCI format for moves
            }
            
            games.append(game_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(games)
    return df

# Usage
pgn_file = 'lichess_db_standard_rated_2015-04.pgn'  # Replace with the path to your PGN file
df = pgn_to_dataframe(pgn_file, num_games=10)  # Adjust num_games as needed

# Display the DataFrame
print(df.head())
