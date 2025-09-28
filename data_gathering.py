import fastf1 as ff1
from fastf1 import ergast
import pandas as pd
from tqdm import tqdm
import traceback
import os # Import the os module

def get_race_data_with_features(year, round_number, ergast_api):
    """Fetches and processes data for a single race, adding advanced features."""
    try:
        session = ff1.get_session(year, round_number, 'R')
        session.load()
        results = session.results
        if results is None or results.empty:
            print(f"No results data found for {year} Round {round_number}.")
            return None

        # Prepare the data
        race_df = results.loc[:, ['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points']].copy()
        race_df.rename(columns={
            'Abbreviation': 'DriverID', 'TeamName': 'TeamID',
            'GridPosition': 'QualifyingPosition', 'Position': 'FinishingPosition'
        }, inplace=True)
        
        race_df['DriverTeamID'] = race_df['DriverID'] + "_" + race_df['TeamID']
        race_df['Year'] = year
        race_df['Round'] = round_number
        race_df['TrackID'] = session.event['Location']

        # Add Championship Standing & Points Features
        standings_before_race = pd.DataFrame()
        if round_number > 1:
            try:
                standings_content = ergast_api.get_driver_standings(season=year, round=round_number - 1).content
                if standings_content:
                    standings_before_race = standings_content[0][['driverCode', 'position', 'points']]
            except Exception as e:
                print(f"  - Could not get standings for {year} R{round_number-1}: {e}")
        
        if not standings_before_race.empty:
            race_df = race_df.merge(standings_before_race, left_on='DriverID', right_on='driverCode', how='left')
            race_df.rename(columns={'position': 'ChampionshipStanding', 'points': 'ChampionshipPoints'}, inplace=True)
            race_df.drop(columns=['driverCode'], inplace=True)
        else:
            race_df['ChampionshipStanding'] = 0
            race_df['ChampionshipPoints'] = 0
        
        race_df.fillna(0, inplace=True)

        # Final data prep
        race_df['FinishingPosition'] = pd.to_numeric(race_df['FinishingPosition'], errors='coerce')
        race_df.dropna(subset=['FinishingPosition'], inplace=True)
        race_df['FinishingPosition'] = race_df['FinishingPosition'].astype(int)
        race_df.loc[race_df['QualifyingPosition'] == 0, 'QualifyingPosition'] = 20
        
        return race_df
    except Exception as e:
        print(f"Could not process data for {year} Round {round_number}. Error: {e}")
        return None

# --- Main Script ---
if __name__ == '__main__':
    # --- START OF THE FIX ---
    # Create the cache directory if it does not exist
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Cache directory '{cache_dir}' created.")
    
    # Now that we know the folder exists, enable the cache
    ff1.Cache.enable_cache(cache_dir)
    print("FastF1 cache enabled successfully.")
    # --- END OF THE FIX ---
    
    ergast_api_main = ergast.Ergast()
    
    ALL_RACE_DATA = []
    # Using years 2023 and 2024 for a solid dataset
    YEARS = [2023, 2024] 
    
    for year in YEARS:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        race_rounds = schedule[schedule['EventName'].str.contains('Grand Prix')]['RoundNumber']
        
        for rnd in tqdm(race_rounds, desc=f"Processing {year} Races"):
            data = get_race_data_with_features(year, rnd, ergast_api_main)
            if data is not None:
                ALL_RACE_DATA.append(data)

    if ALL_RACE_DATA:
        master_df = pd.concat(ALL_RACE_DATA, ignore_index=True)
        
        # Calculate Recent Form features
        master_df.sort_values(by=['Year', 'Round'], inplace=True)
        master_df['RecentFormPoints'] = master_df.groupby('DriverID')['Points'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        ).fillna(0)
        master_df['RecentQualiPos'] = master_df.groupby('DriverID')['QualifyingPosition'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        ).fillna(10) # Fill with a mid-pack default
        master_df['RecentFinishPos'] = master_df.groupby('DriverID')['FinishingPosition'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        ).fillna(10) # Fill with a mid-pack default
        
        # Ensure all necessary columns are selected for the final CSV
        final_columns = [
            'Year', 'Round', 'TrackID', 'DriverID', 'TeamID', 'DriverTeamID', 
            'QualifyingPosition', 'ChampionshipStanding', 'ChampionshipPoints', 'Points',
            'RecentFormPoints', 'RecentQualiPos', 'RecentFinishPos',
            'FinishingPosition'
        ]
        
        master_df = master_df[[col for col in final_columns if col in master_df.columns]]
        
        master_df.to_csv('f1_historical_data.csv', index=False)
        print(f"\n--- Data Gathering Complete! ---")
        print(f"Saved {len(master_df)} records with new 'Form' features to f1_historical_data.csv")
    else:
        print("\n--- No data was gathered. ---")
