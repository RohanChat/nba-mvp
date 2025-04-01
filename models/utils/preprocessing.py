#!/usr/bin/env python3
import argparse
import logging
import os
import sqlite3
import sys
import pandas as pd
import numpy as np

# Set up logging for progress and debugging.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def merge_all_player_stats(pbp_df, player_csv):
    """
    Merge the play-by-play DataFrame with player box stats for all available player columns
    (e.g., player1_id, player2_id, player3_id). For each player column found in pbp_df, the
    function merges the corresponding player stats from player_csv and adds them with a suffix.
    """
    df_player = pd.read_csv(player_csv)
    if 'type' in df_player.columns:
        df_player = df_player.drop(columns=['type'])
    # Convert join keys in the CSV to int.
    df_player['gameid'] = pd.to_numeric(df_player['gameid'], errors='coerce').fillna(0).astype(int)
    df_player['playerid'] = pd.to_numeric(df_player['playerid'], errors='coerce').fillna(0).astype(int)

    # Start with a copy of the play-by-play DataFrame.
    merged_df = pbp_df.copy()
    game_id_col = get_game_id_column(merged_df)
    merged_df[game_id_col] = pd.to_numeric(merged_df[game_id_col], errors='coerce').fillna(0).astype(int)

    # List of expected player columns.
    player_cols = [col for col in ['player1_id', 'player2_id', 'player3_id'] if col in merged_df.columns]
    
    for col in player_cols:
        # Ensure the current player column is numeric.
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
        
        # Save the left join keys so they are not lost after the merge.
        left_game_ids = merged_df[game_id_col].copy()
        left_player_ids = merged_df[col].copy()
        
        # Merge for the current player column.
        merged_player = pd.merge(
            merged_df,
            df_player,
            left_on=[game_id_col, col],
            right_on=['gameid', 'playerid'],
            how='left',
            suffixes=("", f"_{col}")
        )
        # Restore the left join keys.
        merged_player[game_id_col] = left_game_ids
        merged_player[col] = left_player_ids
        # Optionally, drop any duplicate join key columns coming from the right side.
        dup_gameid = f"gameid_{col}"
        dup_playerid = f"playerid_{col}"
        if dup_gameid in merged_player.columns:
            merged_player.drop(columns=[dup_gameid], inplace=True)
        if dup_playerid in merged_player.columns:
            merged_player.drop(columns=[dup_playerid], inplace=True)
        merged_df = merged_player

    return merged_df



def get_play_by_play_in_chunks_csv(game_id, csv_file, chunk_minutes=4):
    """
    Retrieve play-by-play data for a specific game from a CSV file
    and split it into chunks of specified minutes.
    """
    df = pd.read_csv(csv_file)
    if 'gameid' in df.columns:
        game_col = 'gameid'
    else:
        logging.error("No 'gameid' column found in CSV.")
        return None

    # Cast game_id to int if the CSV column is numeric.
    import numpy as np
    if np.issubdtype(df[game_col].dtype, np.number):
        try:
            game_id_val = int(game_id)
        except Exception as e:
            logging.error(f"Could not convert game_id {game_id} to int: {e}")
            return None
    else:
        game_id_val = game_id

    df_game = df[df[game_col] == game_id_val].copy()
    if df_game.empty:
        logging.warning(f"No play-by-play data found for game {game_id_val}")
        return None

    # Parse the ISO 8601 duration in the 'clock' column (e.g., "PT12M00.00S")
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

    if 'clock' in df_game.columns:
        df_game['seconds_remaining'] = df_game['clock'].apply(parse_clock)
    else:
        logging.error("No 'clock' column found in CSV play-by-play data.")
        return None

    period_length = 720  # 12 minutes per period
    df_game['game_seconds'] = (df_game['period'] - 1) * period_length + (period_length - df_game['seconds_remaining'])
    chunk_seconds = chunk_minutes * 60
    df_game['time_chunk'] = (df_game['game_seconds'] // chunk_seconds) + 1

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

    df_game['chunk_label'] = df_game['time_chunk'].apply(lambda x: format_chunk_label(x, chunk_minutes))
    return df_game


def get_play_by_play_in_chunks_sqlite(game_id, sqlite_path, chunk_minutes=4):
    """
    Retrieve play-by-play data for a specific game from a sqlite database
    (table name: play_by_play) and split it into chunks.
    """
    game_id = '00' + str(game_id)  # Ensure game_id is in the correct format.
    conn = sqlite3.connect(sqlite_path)
    query = "SELECT * FROM play_by_play WHERE game_id = ?"
    df_game = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    if df_game.empty:
        logging.warning(f"No play-by-play data found for game {game_id} in sqlite database.")
        return None

    # Use 'pctimestring' (e.g., "12:00") from sqlite.
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

    period_length = 720
    df_game['game_seconds'] = (df_game['period'] - 1) * period_length + (period_length - df_game['seconds_remaining'])
    chunk_seconds = chunk_minutes * 60
    df_game['time_chunk'] = (df_game['game_seconds'] // chunk_seconds) + 1

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

    df_game['chunk_label'] = df_game['time_chunk'].apply(lambda x: format_chunk_label(x, chunk_minutes))
    return df_game


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


def merge_with_team_stats(pbp_df, team_csv):
    """
    Merge the play-by-play DataFrame with team box stats.
    All columns from the team CSV are added (with a suffix) except for 'type'.
    """
    df_team = pd.read_csv(team_csv)
    if 'type' in df_team.columns:
        df_team = df_team.drop(columns=['type'])
    # Rename CSV columns (except join keys) to include a suffix.
    df_team = df_team.rename(columns={col: f"{col}_team" for col in df_team.columns if col not in ['gameid', 'team']})
    
    # Determine join keys from pbp_df.
    game_id_col_pbp = get_game_id_column(pbp_df)  # e.g. 'game_id' or 'gameid'
    team_col_pbp = get_team_column(pbp_df)
    
    # Convert game IDs to int: use errors='coerce' to turn non-numeric into NaN, then fill with 0.
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


def merge_with_player_stats(pbp_df, player_csv):
    """
    Merge the play-by-play DataFrame with player box stats.
    All columns from the player CSV are added (with a suffix) except for 'type'.
    """
    df_player = pd.read_csv(player_csv)
    if 'type' in df_player.columns:
        df_player = df_player.drop(columns=['type'])
    # Rename CSV columns (except join keys) to include a suffix.
    df_player = df_player.rename(columns={col: f"{col}_player" for col in df_player.columns if col not in ['gameid', 'playerid']})
    
    game_id_col_pbp = get_game_id_column(pbp_df)
    player_col_pbp = get_player_column(pbp_df)
    
    # Convert game and player IDs to int: coerce errors then fill missing values with 0.
    pbp_df[game_id_col_pbp] = pd.to_numeric(pbp_df[game_id_col_pbp], errors='coerce').fillna(0).astype(int)
    pbp_df[player_col_pbp] = pd.to_numeric(pbp_df[player_col_pbp], errors='coerce').fillna(0).astype(int)
    df_player['gameid'] = pd.to_numeric(df_player['gameid'], errors='coerce').fillna(0).astype(int)
    df_player['playerid'] = pd.to_numeric(df_player['playerid'], errors='coerce').fillna(0).astype(int)
    
    merged_df = pd.merge(
        pbp_df,
        df_player,
        left_on=[game_id_col_pbp, player_col_pbp],
        right_on=['gameid', 'playerid'],
        how='left'
    )
    return merged_df



def process_single_game(game_id, source, box_team_csv, box_player_csv, chunk_size=4, mode='csv', output_dir='output'):
    """
    Process a single game: chunk the play-by-play data,
    merge with team and player box stats, and write out a CSV.
    """
    logging.info(f"Processing game {game_id} in {mode} mode.")
    if mode == 'csv':
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


def process_year_range(start_year, end_year, mode, chunk_size, box_team_csv, box_player_csv, sqlite_path=None, output_dir='output'):
    """
    Process play-by-play data for all games between start_year and end_year.
    For each year, read the appropriate CSV (or query sqlite), process each game,
    merge with box stats, and write out a CSV for that year.
    
    Additionally, print out the game IDs (and associated teams/players) that did not find a match,
    along with counts.
    """
    logging.info(f"Processing years {start_year} to {end_year} in {mode} mode.")
    if mode == 'sqlite' and sqlite_path is None:
        sqlite_path = "./data/nba_sqlite/nba.sqlite"

    # Dictionaries to track missing matches.
    missing_team_matches = {}    # key: game id, value: list of teams with missing team stats
    missing_player_matches = {}  # key: game id, value: list of players with missing player stats

    processed_games = []
    for year in range(start_year, end_year + 1):
        logging.info(f"Processing year {year}.")
        pbp_csv_file = f"./data/nba_csv/pbp/pbp{year}.csv"
        try:
            df_year = pd.read_csv(pbp_csv_file)
        except Exception as e:
            logging.error(f"Error reading play-by-play file for {year}: {e}")
            continue

        if 'gameid' not in df_year.columns:
            logging.error(f"No 'gameid' column in play-by-play file for year {year}.")
            continue

        game_ids = df_year['gameid'].unique()
        for i, game_id in enumerate(game_ids, 1):
            # Show progress using sys.stdout.
            sys.stdout.write(f"\rProcessing game {i}/{len(game_ids)} (Game ID: {game_id})")
            sys.stdout.flush()

            if mode == 'csv':
                pbp_game_df = get_play_by_play_in_chunks_csv(game_id, pbp_csv_file, chunk_minutes=chunk_size)
            else:
                pbp_game_df = get_play_by_play_in_chunks_sqlite(game_id, sqlite_path, chunk_minutes=chunk_size)

            if pbp_game_df is None:
                logging.warning(f"Skipping game {game_id} for year {year} due to missing pbp data.")
                continue

            # Merge with team and player stats.
            pbp_game_df = merge_with_team_stats(pbp_game_df, box_team_csv)
            pbp_game_df = merge_with_player_stats(pbp_game_df, box_player_csv)

            # --- Missing Match Analysis ---
            # Check team merge: for each unique team, verify that at least one expected team stat column (e.g., 'PTS_team') is non-null.
            team_col = get_team_column(pbp_game_df)
            missing_teams = []
            for team in pbp_game_df[team_col].unique():
                subset = pbp_game_df[pbp_game_df[team_col] == team]
                if 'PTS_team' in subset.columns:
                    if subset['PTS_team'].isna().all():
                        missing_teams.append(team)
                else:
                    # Fallback: look for any column ending with '_team'
                    candidate = [col for col in subset.columns if col.endswith('_team')]
                    if candidate and subset[candidate[0]].isna().all():
                        missing_teams.append(team)
            if missing_teams:
                missing_team_matches[game_id] = missing_teams

            # Check player merge: for each unique player (skip 0 or NaN), verify an expected player stat column (e.g., 'PTS_player').
            player_col = get_player_column(pbp_game_df)
            missing_players = []
            for player in pbp_game_df[player_col].unique():
                try:
                    if pd.isna(player) or int(player) == 0:
                        continue
                except Exception:
                    continue
                subset = pbp_game_df[pbp_game_df[player_col] == player]
                if 'PTS_player' in subset.columns:
                    if subset['PTS_player'].isna().all():
                        missing_players.append(player)
                else:
                    candidate = [col for col in subset.columns if col.endswith('_player')]
                    if candidate and subset[candidate[0]].isna().all():
                        missing_players.append(player)
            if missing_players:
                missing_player_matches[game_id] = missing_players

            processed_games.append(pbp_game_df)
        sys.stdout.write("\n")  # Move to next line after progress

        if processed_games:
            df_processed_year = pd.concat(processed_games, ignore_index=True)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"processed_{year}.csv")
            df_processed_year.to_csv(output_file, index=False)
            logging.info(f"Year {year}: processed {len(processed_games)} games saved to {output_file}.")
        else:
            logging.warning(f"No games processed for year {year}.")

    # After processing all years, print analysis of missing matches.
    total_games = len(game_ids)
    missing_team_count = len(missing_team_matches)
    missing_player_count = len(missing_player_matches)
    logging.info(f"\nMissing Team Stats: {missing_team_count} out of {total_games} games had missing team matches.")
    logging.info(f"Missing Player Stats: {missing_player_count} out of {total_games} games had missing player matches.")
    if missing_team_matches:
        logging.info("Game IDs with missing team stats and their missing teams:")
        for gid, teams in missing_team_matches.items():
            logging.info(f"Game ID {gid}: missing teams: {teams}")
    if missing_player_matches:
        logging.info("Game IDs with missing player stats and their missing players:")
        for gid, players in missing_player_matches.items():
            logging.info(f"Game ID {gid}: missing players: {players}")


def main():
    parser = argparse.ArgumentParser(description="NBA Play-by-Play Preprocessing Script")
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Choose 'single' to process one game or 'yearrange' to process multiple years.")

    # Sub-parser for single game processing.
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

    # Sub-parser for processing a range of years.
    yr_parser = subparsers.add_parser("yearrange", help="Process a range of years")
    yr_parser.add_argument("--start_year", type=int, required=True, help="Start year")
    yr_parser.add_argument("--end_year", type=int, required=True, help="End year")
    yr_parser.add_argument("--mode", type=str, choices=['csv', 'sqlite'], default='csv',
                           help="Data source mode: 'csv' or 'sqlite'")
    yr_parser.add_argument("--chunk_size", type=int, default=4, help="Chunk size in minutes (default: 4)")
    yr_parser.add_argument("--box_team_csv", type=str, default="./data/nba_csv/team_traditional.csv",
                           help="Path to team box stats CSV (default: ./data/nba_csv/team_traditional.csv)")
    yr_parser.add_argument("--box_player_csv", type=str, default="./data/nba_csv/traditional.csv",
                           help="Path to player box stats CSV (default: ./data/nba_csv/traditional.csv)")
    yr_parser.add_argument("--sqlite_path", type=str, default="./data/nba_sqlite/nba.sqlite",
                           help="Path to sqlite database (default: ./data/nba_sqlite/nba.sqlite)")
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
