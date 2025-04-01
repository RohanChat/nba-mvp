#!/usr/bin/env python3
import argparse
import logging
import os
import sqlite3
import sys
import pandas as pd
import numpy as np
import concurrent.futures

# Set up logging for progress and debugging.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- New helper functions for in-memory processing ---

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
    Merge the play-by-play DataFrame with player box stats for all available player columns
    (e.g., player1_id, player2_id, player3_id). For each player column found in pbp_df, the
    function merges the corresponding player stats from player_csv and adds them with a suffix.
    """
    df_player = pd.read_csv(player_csv)
    if 'type' in df_player.columns:
        df_player = df_player.drop(columns=['type'])
    df_player['gameid'] = pd.to_numeric(df_player['gameid'], errors='coerce').fillna(0).astype(int)
    df_player['playerid'] = pd.to_numeric(df_player['playerid'], errors='coerce').fillna(0).astype(int)

    merged_df = pbp_df.copy()
    game_id_col = get_game_id_column(merged_df)
    merged_df[game_id_col] = pd.to_numeric(merged_df[game_id_col], errors='coerce').fillna(0).astype(int)

    player_cols = [col for col in ['player1_id', 'player2_id', 'player3_id'] if col in merged_df.columns]
    
    for col in player_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
        left_game_ids = merged_df[game_id_col].copy()
        left_player_ids = merged_df[col].copy()
        merged_player = pd.merge(
            merged_df,
            df_player,
            left_on=[game_id_col, col],
            right_on=['gameid', 'playerid'],
            how='left',
            suffixes=("", f"_{col}")
        )
        merged_player[game_id_col] = left_game_ids
        merged_player[col] = left_player_ids
        dup_gameid = f"gameid_{col}"
        dup_playerid = f"playerid_{col}"
        if dup_gameid in merged_player.columns:
            merged_player.drop(columns=[dup_gameid], inplace=True)
        if dup_playerid in merged_player.columns:
            merged_player.drop(columns=[dup_playerid], inplace=True)
        merged_df = merged_player

    return merged_df

def merge_with_team_stats(pbp_df, team_csv):
    """
    Merge the play-by-play DataFrame with team box stats.
    All columns from the team CSV are added (with a suffix) except for 'type'.
    """
    df_team = pd.read_csv(team_csv)
    if 'type' in df_team.columns:
        df_team = df_team.drop(columns=['type'])
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
        # Adjust this path as needed for Deepnote
        pbp_csv_file = f"/datasets/_deepnote_work/data/nba_csv/pbp/pbp{year}.csv"
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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit a job for each game.
            futures = {executor.submit(process_game, game_id, df_year, chunk_size, box_team_csv, box_player_csv): game_id for game_id in game_ids}
            for future in concurrent.futures.as_completed(futures):
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
