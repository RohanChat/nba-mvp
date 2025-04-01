#!/usr/bin/env python3
import argparse
import logging
import os
import sqlite3
import sys
import pandas as pd
import numpy as np
import concurrent.futures
from preprocessing import get_play_by_play_in_chunks_csv, get_play_by_play_in_chunks_sqlite

# Set up logging for progress and debugging.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- New helper functions for in-memory processing ---

def process_game_sqlite(game_id, sqlite_path, chunk_size, box_team_csv, box_player_csv):
    """
    Process a single game from the sqlite database.
    Opens a connection, retrieves play-by-play data for the given game,
    applies chunking, and then merges team and player stats.
    """
    # Format game_id if necessary (e.g., padding with zeros)
    formatted_game_id = str(game_id).zfill(10)  # adjust based on your game_id format

    conn = sqlite3.connect(sqlite_path)
    query = "SELECT * FROM play_by_play WHERE game_id = ?"
    df_game = pd.read_sql_query(query, conn, params=(formatted_game_id,))
    conn.close()
    
    if df_game.empty:
        logging.warning(f"No play-by-play data found for game {game_id} in sqlite.")
        return None

    # Use the sqlite-specific chunking logic.
    def parse_pctimestring(time_str):
        try:
            minutes, seconds = time_str.split(":")
            return int(minutes) * 60 + int(seconds)
        except Exception:
            return np.nan

    if 'pctimestring' in df_game.columns:
        df_game['seconds_remaining'] = df_game['pctimestring'].apply(parse_pctimestring)
    else:
        logging.error("No 'pctimestring' column found in sqlite play-by-play data.")
        return None

    period_length = 720  # 12 minutes
    df_game['game_seconds'] = (df_game['period'] - 1) * period_length + (period_length - df_game['seconds_remaining'])
    chunk_seconds = chunk_size * 60
    df_game['time_chunk'] = (df_game['game_seconds'] // chunk_seconds) + 1

    # Here you can either re-use the chunk labeling logic from your CSV function
    # or simplify it; for example:
    df_game['chunk_label'] = df_game['time_chunk'].apply(lambda x: f"Chunk {x}")

    # Merge team and player stats.
    df_game = merge_with_team_stats(df_game, box_team_csv)
    df_game = merge_all_player_stats(df_game, box_player_csv)
    
    return df_game


def chunk_game_df(game_df, chunk_minutes=4):
    """
    Given a DataFrame containing play-by-play data for a single game,
    parse the clock, compute elapsed seconds and assign a chunk label.
    """
    def parse_clock(clock_str):
        if pd.isna(clock_str):
            return np.nan
        clock_str = clock_str.strip()
        if not clock_str.startswith("PT"):
            return np.nan
        clock_str = clock_str[2:]
        if "M" in clock_str:
            parts = clock_str.split("M")
            try:
                minutes = int(parts[0])
            except Exception:
                minutes = 0
            seconds_str = parts[1].rstrip("S")
            try:
                seconds = float(seconds_str) if seconds_str != "" else 0.0
            except Exception:
                seconds = 0.0
            return minutes * 60 + seconds
        else:
            seconds_str = clock_str.rstrip("S")
            try:
                return float(seconds_str)
            except Exception:
                return np.nan

    if 'clock' not in game_df.columns:
        logging.error("No 'clock' column found in game DataFrame.")
        return None
    game_df['seconds_remaining'] = game_df['clock'].apply(parse_clock)
    period_length = 720  # 12 minutes per period
    game_df['game_seconds'] = (game_df['period'] - 1) * period_length + (period_length - game_df['seconds_remaining'])
    chunk_seconds = chunk_minutes * 60
    game_df['time_chunk'] = (game_df['game_seconds'] // chunk_seconds) + 1

    def format_chunk_label(chunk_num, chunk_minutes):
        start_min = (chunk_num - 1) * chunk_minutes
        end_min = chunk_num * chunk_minutes
        start_qtr = start_min // 12 + 1
        start_qtr_min = start_min % 12
        end_qtr = end_min // 12 + 1
        end_qtr_min = end_min % 12
        if start_qtr == end_qtr:
            return f"Q{start_qtr}: {start_qtr_min}-{end_qtr_min} min"
        else:
            return f"Q{start_qtr}: {start_qtr_min} min - Q{end_qtr}: {end_qtr_min} min"
    game_df['chunk_label'] = game_df['time_chunk'].apply(lambda x: format_chunk_label(x, chunk_minutes))
    return game_df


def process_game(game_id, df_year, chunk_size, box_team_csv, box_player_csv):
    """
    Process a single game from an in-memory DataFrame (df_year) by:
      - Filtering the game,
      - Chunking its play-by-play data,
      - Merging team and all player stats.
    Returns the processed DataFrame.
    """
    game_df = df_year[df_year['gameid'] == game_id].copy()
    if game_df.empty:
        logging.warning(f"No play-by-play data for game {game_id}")
        return None
    # Chunk the game data.
    game_df = chunk_game_df(game_df, chunk_minutes=chunk_size)
    if game_df is None:
        return None
    # Merge team stats.
    game_df = merge_with_team_stats(game_df, box_team_csv)
    # Merge all player stats (merging home, visitor, and neutral players).
    game_df = merge_all_player_stats(game_df, box_player_csv)
    return game_df

# --- Existing merge functions (unchanged, or as previously fixed) ---

def merge_all_player_stats(pbp_df, player_csv):
    """
    Merge the play-by-play DataFrame with player box stats from traditional.csv.
    If pbp_df has multiple player columns (player1_id, etc.), it merges each.
    For the primary player's stats (from player1_id), the suffix will be "_player"
    so that, for example, a points column becomes "PTS_player". Other player columns
    will receive a suffix based on their column name.
    The game id and player id columns from pbp_df are preserved.
    """
    df_player = pd.read_csv(player_csv)
    if 'type' in df_player.columns:
        df_player = df_player.drop(columns=['type'])
    # Convert join keys in the player CSV to int.
    df_player['gameid'] = pd.to_numeric(df_player['gameid'], errors='coerce').fillna(0).astype(int)
    df_player['playerid'] = pd.to_numeric(df_player['playerid'], errors='coerce').fillna(0).astype(int)

    game_id_col = get_game_id_column(pbp_df)
    pbp_df[game_id_col] = pd.to_numeric(pbp_df[game_id_col], errors='coerce').fillna(0).astype(int)

    if 'player1_id' in pbp_df.columns:
        # If there are multiple player columns, process each.
        for col in ['player1_id', 'player2_id', 'player3_id']:
            if col in pbp_df.columns:
                pbp_df[col] = pd.to_numeric(pbp_df[col], errors='coerce').fillna(0).astype(int)
                left_game_ids = pbp_df[game_id_col].copy()
                left_player_ids = pbp_df[col].copy()
                # For the primary player's stats, use the suffix "_player"
                suffix = "_player" if col == "player1_id" else f"_{col}"
                merged = pd.merge(
                    pbp_df,
                    df_player,
                    left_on=[game_id_col, col],
                    right_on=['gameid', 'playerid'],
                    how='left',
                    suffixes=("", suffix)
                )
                # Drop duplicate join key columns that came from the right side,
                # but only those that include our suffix.
                if f"gameid{suffix}" in merged.columns:
                    merged.drop(columns=[f"gameid{suffix}"], inplace=True)
                if f"playerid{suffix}" in merged.columns:
                    merged.drop(columns=[f"playerid{suffix}"], inplace=True)
                # Restore the left-hand join keys
                merged[game_id_col] = left_game_ids
                merged[col] = left_player_ids
                pbp_df = merged
        return pbp_df
    elif 'playerid' in pbp_df.columns:
        # For CSV play-by-play files with a single 'playerid' column.
        pbp_df['playerid'] = pd.to_numeric(pbp_df['playerid'], errors='coerce').fillna(0).astype(int)
        merged = pd.merge(
            pbp_df,
            df_player,
            left_on=[game_id_col, 'playerid'],
            right_on=['gameid', 'playerid'],
            how='left',
            suffixes=("", "_player")
        )
        # Drop duplicate join key columns that include the suffix.
        if "gameid_player" in merged.columns:
            merged.drop(columns=["gameid_player"], inplace=True)
        return merged
    else:
        logging.error("No player id column found in play-by-play data.")
        return pbp_df




def merge_with_team_stats(pbp_df, team_csv):
    """
    Merge the play-by-play DataFrame with team box stats.
    """
    df_team = pd.read_csv(team_csv)
    df_team = df_team.rename(columns={col: f"{col}_team" for col in df_team.columns if col not in ['gameid', 'team']})
    game_id_col_pbp = get_game_id_column(pbp_df)
    team_col_pbp = get_team_column(pbp_df)
    pbp_df[game_id_col_pbp] = pd.to_numeric(pbp_df[game_id_col_pbp], errors='coerce').fillna(0).astype(int)
    df_team['gameid'] = pd.to_numeric(df_team['gameid'], errors='coerce').fillna(0).astype(int)
    merged_df = pd.merge(
        pbp_df,
        df_team,
        left_on=[game_id_col_pbp, team_col_pbp],
        right_on=['gameid', 'team'],
        how='left'
    )
    return merged_df

def get_game_id_column(df):
    if 'gameid' in df.columns:
        return 'gameid'
    elif 'game_id' in df.columns:
        return 'game_id'
    else:
        raise ValueError("No game id column found in the DataFrame.")

def get_game_ids_sqlite(sqlite_path):
    """Retrieve all distinct game IDs from the sqlite play_by_play table."""
    conn = sqlite3.connect(sqlite_path)
    query = "SELECT DISTINCT game_id FROM play_by_play"
    df_ids = pd.read_sql_query(query, conn)
    conn.close()
    # Optionally, if your game_id is stored as text but you want int, you can do:
    df_ids['game_id'] = pd.to_numeric(df_ids['game_id'], errors='coerce').fillna(0).astype(int)
    return df_ids['game_id'].unique()


def get_team_column(df):
    if 'team' in df.columns:
        return 'team'
    elif 'player1_team_abbreviation' in df.columns:
        return 'player1_team_abbreviation'
    else:
        raise ValueError("No team column found in the DataFrame.")

def get_player_column(df):
    if 'playerid' in df.columns:
        return 'playerid'
    elif 'player1_id' in df.columns:
        return 'player1_id'
    else:
        raise ValueError("No player id column found in the DataFrame.")

# --- Updated process_year_range using parallel processing and in-memory filtering ---

def process_year_range(start_year, end_year, mode, chunk_size, box_team_csv, box_player_csv, sqlite_path=None, output_dir='output'):
    logging.info(f"Processing years {start_year} to {end_year} in {mode} mode.")
    if mode != 'csv':
        logging.error("This optimized version currently supports CSV mode only.")
        return

    for year in range(start_year, end_year + 1):
        pbp_csv_file = f"./data/nba_csv/pbp/pbp{year}.csv"
        logging.info(f"Processing year {year}.")
        try:
            df_year = pd.read_csv(pbp_csv_file)
        except Exception as e:
            logging.error(f"Error reading play-by-play file for {year}: {e}")
            continue

        if 'gameid' not in df_year.columns:
            logging.error(f"No 'gameid' column in play-by-play file for year {year}.")
            continue

        # Convert key columns once
        df_year['gameid'] = pd.to_numeric(df_year['gameid'], errors='coerce').fillna(0).astype(int)
        for col in ['player1_id', 'player2_id', 'player3_id']:
            if col in df_year.columns:
                df_year[col] = pd.to_numeric(df_year[col], errors='coerce').fillna(0).astype(int)

        game_ids = df_year['gameid'].unique()
        logging.info(f"Year {year}: found {len(game_ids)} games.")

        results = []
        total_games = len(game_ids)
        processed_count = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit a job for each game.
            futures = {executor.submit(process_game, game_id, df_year, chunk_size, box_team_csv, box_player_csv): game_id for game_id in game_ids}
            for future in concurrent.futures.as_completed(futures):
                processed_count += 1
                sys.stdout.write(f"\rProcessing game {processed_count}/{total_games}...")
                sys.stdout.flush()
                game_id = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    logging.error(f"Error processing game {game_id}: {e}")

        if results:
            df_processed_year = pd.concat(results, ignore_index=True)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"processed_{year}.csv")
            df_processed_year.to_csv(output_file, index=False)
            logging.info(f"Year {year}: processed {len(results)} games saved to {output_file}.")
        else:
            logging.warning(f"No games processed for year {year}.")

def process_year_range_sqlite(start_year, end_year, sqlite_path, chunk_size, box_team_csv, box_player_csv, output_dir='output'):
    logging.info(f"Processing sqlite data for years {start_year} to {end_year} in parallel mode.")
    overall_missing_games = {}
    all_results = []

    for year in range(start_year, end_year + 1):
        # Load the CSV file for this year (adjust path as needed)
        pbp_csv_file = f"./data/nba_csv/pbp/pbp{year}.csv"
        try:
            df_year = pd.read_csv(pbp_csv_file)
        except Exception as e:
            logging.error(f"Error reading play-by-play CSV for year {year}: {e}")
            continue

        if 'gameid' not in df_year.columns:
            logging.error(f"No 'gameid' column in CSV for year {year}.")
            continue

        # Get game IDs from CSV file (this restricts to only the games for this year)
        game_ids = pd.to_numeric(df_year['gameid'], errors='coerce').dropna().astype(int).unique()
        logging.info(f"Year {year}: found {len(game_ids)} games in CSV.")

        results = []
        missing_games = []
        total_games = len(game_ids)
        processed_count = 0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_game_sqlite, game_id, sqlite_path, chunk_size, box_team_csv, box_player_csv): game_id
                for game_id in game_ids
            }
            for future in concurrent.futures.as_completed(futures):
                processed_count += 1
                sys.stdout.write(f"\rYear {year}: Processing game {processed_count}/{total_games}...")
                sys.stdout.flush()
                game_id = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    else:
                        # Record missing game if the sqlite query returned no data.
                        missing_games.append(game_id)
                except Exception as e:
                    logging.error(f"Error processing game {game_id}: {e}")
            sys.stdout.write("\n")  # New line after progress

        # Save missing game info for this year.
        if missing_games:
            overall_missing_games[year] = missing_games
            logging.warning(f"Year {year}: {len(missing_games)} games had no matching data in sqlite.")

        if results:
            df_processed_year = pd.concat(results, ignore_index=True)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"processed_sqlite_{year}.csv")
            df_processed_year.to_csv(output_file, index=False)
            logging.info(f"Year {year}: processed {len(results)} games saved to {output_file}.")
            all_results.append(df_processed_year)
        else:
            logging.warning(f"No games processed for year {year}.")

    # Optionally, save a summary of missing games.
    if overall_missing_games:
        missing_file = os.path.join(output_dir, "missing_games_by_year.txt")
        with open(missing_file, "w") as f:
            for year, games in overall_missing_games.items():
                f.write(f"Year {year}: {len(games)} missing games - {games}\n")
        logging.info(f"Missing game IDs written to {missing_file}.")
    
    # Optionally, combine all years into one CSV.
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(output_dir, "processed_sqlite_all_years.csv")
        df_all.to_csv(combined_file, index=False)
        logging.info(f"All years combined processed data saved to {combined_file}.")



def process_single_game(game_id, source, box_team_csv, box_player_csv, chunk_size=4, mode='csv', output_dir='output'):
    logging.info(f"Processing game {game_id} in {mode} mode.")
    if mode == 'csv':
        # In single-game mode, we can simply load the file and filter.
        pbp_df = get_play_by_play_in_chunks_csv(game_id, source, chunk_minutes=chunk_size)
    elif mode == 'sqlite':
        pbp_df = get_play_by_play_in_chunks_sqlite(game_id, source, chunk_minutes=chunk_size)
    else:
        logging.error("Invalid mode. Please choose 'csv' or 'sqlite'.")
        return

    if pbp_df is None:
        logging.error(f"Failed to process play-by-play data for game {game_id}.")
        return

    pbp_df = merge_with_team_stats(pbp_df, box_team_csv)
    pbp_df = merge_all_player_stats(pbp_df, box_player_csv)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"processed_game_{game_id}.csv")
    pbp_df.to_csv(output_file, index=False)
    logging.info(f"Processed game {game_id} saved to {output_file}.")
    return pbp_df

def main():
    parser = argparse.ArgumentParser(description="NBA Play-by-Play Preprocessing Script")
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Choose 'single' to process one game or 'yearrange' to process multiple years.")

    single_parser = subparsers.add_parser("single", help="Process a single game")
    single_parser.add_argument("--game_id", type=str, required=True, help="Game ID to process")
    single_parser.add_argument("--source", type=str, required=True,
                               help="CSV file (if mode=csv) or sqlite DB (if mode=sqlite) containing play-by-play data")
    single_parser.add_argument("--box_team_csv", type=str, required=True, help="Path to team box stats CSV")
    single_parser.add_argument("--box_player_csv", type=str, required=True, help="Path to player box stats CSV")
    single_parser.add_argument("--chunk_size", type=int, default=4, help="Chunk size in minutes (default: 4)")
    single_parser.add_argument("--mode", type=str, choices=['csv', 'sqlite'], default='csv',
                               help="Data source mode: 'csv' or 'sqlite'")
    single_parser.add_argument("--output_dir", type=str, default="output", help="Output directory for CSV file")

    yr_parser = subparsers.add_parser("yearrange", help="Process a range of years")
    yr_parser.add_argument("--start_year", type=int, required=True, help="Start year")
    yr_parser.add_argument("--end_year", type=int, required=True, help="End year")
    yr_parser.add_argument("--mode", type=str, choices=['csv', 'sqlite'], default='csv',
                           help="Data source mode: 'csv' or 'sqlite'")
    yr_parser.add_argument("--chunk_size", type=int, default=4, help="Chunk size in minutes (default: 4)")
    yr_parser.add_argument("--box_team_csv", type=str, required=True,
                           help="Path to team box stats CSV")
    yr_parser.add_argument("--box_player_csv", type=str, required=True,
                           help="Path to player box stats CSV")
    yr_parser.add_argument("--sqlite_path", type=str, default="./data/nba_sqlite/nba.sqlite",
                           help="Path to sqlite database (if mode=sqlite)")
    yr_parser.add_argument("--output_dir", type=str, default="output", help="Output directory for CSV files")

    args = parser.parse_args()

    if args.command == "single":
        process_single_game(
            game_id=args.game_id,
            source=args.source,
            box_team_csv=args.box_team_csv,
            box_player_csv=args.box_player_csv,
            chunk_size=args.chunk_size,
            mode=args.mode,
            output_dir=args.output_dir
        )
    elif args.command == "yearrange":
        if args.mode == "sqlite":
            process_year_range_sqlite(
                start_year=args.start_year,
                end_year=args.end_year,
                sqlite_path=args.sqlite_path,
                chunk_size=args.chunk_size,
                box_team_csv=args.box_team_csv,
                box_player_csv=args.box_player_csv,
                output_dir=args.output_dir
            )
        else:
            process_year_range(
                start_year=args.start_year,
                end_year=args.end_year,
                mode=args.mode,
                chunk_size=args.chunk_size,
                box_team_csv=args.box_team_csv,
                box_player_csv=args.box_player_csv,
                sqlite_path=args.sqlite_path,
                output_dir=args.output_dir
            )


if __name__ == "__main__":
    main()
