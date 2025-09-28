import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import fastf1 as ff1
from fastf1 import ergast
import fastf1.plotting as ff1_plt 
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import traceback 
import requests
from functools import lru_cache
from tqdm import tqdm
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

MY_COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFF200",
    "HARD": "#EBEBEB",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
}

# --- API Key and Global Configs ---


# --- NEW: OPENWEATHERMAP API KEY ---
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

if OPENWEATHER_API_KEY == "YOUR_NEW_OPENWEATHERMAP_API_KEY_HERE" or OPENWEATHER_API_KEY == "":
    print("Warning: OPENWEATHER_API_KEY is not set. Weather features will be disabled.")
    OPENWEATHER_API_KEY = None


ERGAS_CONSTRUCTOR_NAME_TO_KNOWN_TEAM_ID_OR_COLOR = {
    "Red Bull": "#0600EF", "Ferrari": "#E80020", "Mercedes": "#27F4D2",
    "McLaren": "#FF8000", "Aston Martin": "#229971", "Sauber": "#00FF00", 
    "Haas F1 Team": "Haas", "RB F1 Team": "#00359F", "Williams": "#64C4FF",
    "Alpine F1 Team":  "#0090FF", "Alpine": "#0090FF" 
}

# Dictionary 1: Maps TeamName from FastF1 session results to the official ConstructorName from Ergast API
TEAM_NAME_TO_CONSTRUCTOR_NAME_MAP = {
    "Red Bull Racing": "Red Bull",
    "Mercedes": "Mercedes",
    "Ferrari": "Ferrari",
    "McLaren": "McLaren",
    "Aston Martin": "Aston Martin",
    "Alpine": "Alpine F1 Team",  # Map the name we see in results to the name in standings
    "Racing Bulls": "RB F1 Team", # Map the name we see in results to the name in standings
    "Sauber": "Sauber", 
    "Kick Sauber": "Sauber",     # Also map "Kick Sauber" to the base "Sauber" name from standings
    "Haas F1 Team": "Haas F1 Team",
    "Williams": "Williams",
}

# Dictionary 2: Maps the official ConstructorName to a specific color
CONSTRUCTOR_NAME_TO_COLOR_MAP = {
    "Red Bull": "#0600EF",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine F1 Team": "#0090FF",
    "RB F1 Team": "#00359F",
    "Sauber": "#00FF00",
    "Haas F1 Team": "#B6BABD",
    "Williams": "#64C4FF",
}

# --- 1. Configure FastF1 Caching ---
cache_dir = 'cache'
# Create the cache directory on the server if it does not exist
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Cache directory '{cache_dir}' created.")

# Now that we know the folder exists, enable the cache
try:
    ff1.Cache.enable_cache(cache_dir)
    print("FastF1 cache enabled successfully.")
except Exception as e:
    print(f"An unexpected error occurred while enabling FastF1 cache: {e}")

try:
    circuit_data_df = pd.read_csv("circuit_data.csv")
    print("Successfully loaded circuit_data.csv")
except FileNotFoundError:
    print("Warning: circuit_data.csv not found. Circuit specific data will be missing.")
    circuit_data_df = pd.DataFrame() # Create an empty DataFrame if file is missing

# --- HELPER FUNCTION DEFINITIONS ---
def format_timedelta(td_object):
    if pd.isna(td_object): return "N/A"
    if isinstance(td_object, pd.Timedelta):
        total_seconds = td_object.total_seconds(); sign = "-" if total_seconds < 0 else ""; total_seconds = abs(total_seconds)
        hours = int(total_seconds // 3600); minutes = int((total_seconds % 3600) // 60); seconds = total_seconds % 60
        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:06.3f}" if hours > 0 else f"{sign}{minutes:02d}:{seconds:06.3f}"
    return str(td_object)

def hex_to_rgba(hex_color, alpha=0.15):
    """Converts a hex color string to an rgba string with a given alpha."""
    try:
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    except:
        # Fallback if the color is not a valid hex
        return None 

@lru_cache(maxsize=4) # Cache the result of this very slow function

# --- ADD THIS HELPER FUNCTION to app.py ---
# (Place it before the run_reinforcement_simulation function)

# --- REPLACE the existing get_race_data function in app.py with this one ---
def get_features_for_race(year, round_number, ergast_api):
    """Fetches and processes data for a single race, adding advanced features."""
    try:
        session = ff1.get_session(year, round_number, 'R')
        session.load()
        results = session.results
        if results is None or results.empty: return None

        race_df = results.loc[:, ['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points']].copy()
        race_df.rename(columns={'Abbreviation': 'DriverID', 'TeamName': 'TeamID', 'GridPosition': 'QualifyingPosition', 'Position': 'FinishingPosition'}, inplace=True)
        
        race_df['DriverTeamID'] = race_df['DriverID'] + "_" + race_df['TeamID']
        race_df['Year'] = year
        race_df['Round'] = round_number
        race_df['TrackID'] = session.event['Location']

        standings_before_race = pd.DataFrame()
        if round_number > 1:
            try:
                standings_content = ergast_api.get_driver_standings(season=year, round=round_number - 1).content
                if standings_content: standings_before_race = standings_content[0][['driverCode', 'position', 'points']]
            except Exception as e: print(f"  - Could not get standings for {year} R{round_number-1}: {e}")
        
        if not standings_before_race.empty:
            race_df = race_df.merge(standings_before_race, left_on='DriverID', right_on='driverCode', how='left')
            race_df.rename(columns={'position': 'ChampionshipStanding', 'points': 'ChampionshipPoints'}, inplace=True)
            race_df.drop(columns=['driverCode'], inplace=True)
        else:
            race_df['ChampionshipStanding'] = 0
            race_df['ChampionshipPoints'] = 0
        
        race_df.fillna(0, inplace=True)
        race_df['FinishingPosition'] = pd.to_numeric(race_df['FinishingPosition'], errors='coerce')
        race_df.dropna(subset=['FinishingPosition'], inplace=True)
        race_df['FinishingPosition'] = race_df['FinishingPosition'].astype(int)
        race_df.loc[race_df['QualifyingPosition'] == 0, 'QualifyingPosition'] = 20
        
        return race_df
    except Exception as e:
        print(f"Could not process data for {year} Round {round_number}. Error: {e}")
        return None

def get_race_data_with_features(year, round_number, ergast_api):
    """Fetches and processes data for a single race, adding advanced features."""
    try:
        session = ff1.get_session(year, round_number, 'R')
        session.load()
        results = session.results
        if results is None or results.empty: return None

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
                if standings_content: standings_before_race = standings_content[0][['driverCode', 'position', 'points']]
            except Exception as e: print(f"  - Could not get standings for {year} R{round_number-1}: {e}")
        
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

@lru_cache(maxsize=4)
def run_reinforcement_simulation(year, historical_data_path, initial_model_path):
    print(f"\n[Reinforcement Sim] Starting for year {year}...")
    try:
        base_df = pd.read_csv(historical_data_path)
        base_model = joblib.load(initial_model_path)
        
        schedule = ff1.get_event_schedule(year, include_testing=False)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()].sort_values(by='RoundNumber')

        if completed_races.empty:
            print("[Reinforcement Sim] No completed races yet for this season.")
            return None

        ergast_api_main = ergast.Ergast()
        simulation_results = []
        current_training_data = base_df.copy()

        for _, race in tqdm(completed_races.iterrows(), total=len(completed_races), desc=f"Simulating {year} season"):
            round_num = race['RoundNumber']
            new_race_data = get_features_for_race(year, round_num, ergast_api_main)
            if new_race_data is None or new_race_data.empty: continue

            # --- Calculate "form" features for the race we are about to predict ---
            form_data = []
            for driver_id in new_race_data['DriverID']:
                driver_history = current_training_data[current_training_data['DriverID'] == driver_id]
                last_5 = driver_history.tail(5)
                form_data.append({
                    'DriverID': driver_id,
                    'RecentFormPoints': last_5['Points'].mean() if not last_5.empty else 0,
                    'RecentQualiPos': last_5['QualifyingPosition'].mean() if not last_5.empty else 10,
                    'RecentFinishPos': last_5['FinishingPosition'].mean() if not last_5.empty else 10
                })
            form_df = pd.DataFrame(form_data)
            new_race_data = new_race_data.merge(form_df, on='DriverID', how='left').fillna(0)

            # --- Prepare full training data (up to the previous race) ---
            X_train = current_training_data.drop(columns=['FinishingPosition', 'DriverID', 'TeamID', 'Points'])
            X_train_cat = pd.get_dummies(X_train[['DriverTeamID', 'TrackID']])
            X_train = pd.concat([X_train.drop(columns=['DriverTeamID', 'TrackID']), X_train_cat], axis=1)
            X_train.columns = [str(col) for col in X_train.columns]
            y_train = current_training_data['FinishingPosition']

            # Train a model on all data available *before* this race
            current_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            current_model.fit(X_train, y_train)
            
            # Prepare data for the current race for prediction
            X_predict = new_race_data.drop(columns=['FinishingPosition', 'DriverID', 'TeamID', 'Points'])
            X_predict_cat = pd.get_dummies(X_predict[['DriverTeamID', 'TrackID']])
            X_predict = pd.concat([X_predict.drop(columns=['DriverTeamID', 'TrackID']), X_predict_cat], axis=1)
            
            X_predict_aligned = X_predict.reindex(columns=current_model.get_booster().feature_names, fill_value=0)
            
            predictions = current_model.predict(X_predict_aligned)
            new_race_data['PredictedPosition'] = predictions
            
            # Store results
            predicted_top_10 = new_race_data.sort_values(by='PredictedPosition').head(10)['DriverID'].tolist()
            actual_top_10 = new_race_data.sort_values(by='FinishingPosition').head(10)['DriverID'].tolist()
            
            simulation_results.append({
                "Round": round_num, "Race": race['EventName'],
                "PredictedTop10": predicted_top_10, "ActualTop10": actual_top_10,
                "MAE": mean_absolute_error(new_race_data['FinishingPosition'], predictions),
                "FeatureImportances": dict(zip(current_model.get_booster().feature_names, current_model.feature_importances_))
            })

            # "Reinforce": Add this race's data to the training set for the next iteration
            current_training_data = pd.concat([current_training_data, new_race_data], ignore_index=True)

        return pd.DataFrame(simulation_results)
    except Exception as e:
        print(f"Error during reinforcement simulation: {e}")
        import traceback; traceback.print_exc()
        return None

# REPLACE your get_weather_forecast function with this OpenWeatherMap version
def get_weather_forecast(lat, lon, api_key, race_date_obj):
    if not api_key or api_key == "YOUR_OPENWEATHER_API_KEY_HERE":
        print("[Weather] OpenWeather API key not provided. Skipping forecast fetch.")
        return []
    
    try:
        # OpenWeatherMap One Call API provides an 8-day daily forecast, which is perfect
        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={api_key}&units=metric"
        
        print(f"[Weather] Fetching forecast from OpenWeatherMap...")
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        
        weather_data = response.json()
        forecast_list = []
        
        # Determine the dates for Friday, Saturday, and Sunday of the race weekend
        race_sunday = race_date_obj
        race_saturday = race_sunday - pd.Timedelta(days=1)
        race_friday = race_sunday - pd.Timedelta(days=2)
        race_weekend_dates = [race_friday.date(), race_saturday.date(), race_sunday.date()]
        
        # Go through the daily forecast and find the matching days
        for day_data in weather_data.get('daily', []):
            forecast_date = datetime.fromtimestamp(day_data['dt']).date()
            
            if forecast_date in race_weekend_dates:
                # Format the date to show the day of the week (e.g., "Friday")
                day_name = datetime.fromtimestamp(day_data['dt']).strftime('%A')
                
                forecast = {
                    "Date": day_name, # Show "Friday", "Saturday", "Sunday"
                    "Temp": f"{day_data.get('temp', {}).get('day', 'N/A'):.0f}°C",
                    "Description": day_data.get('weather', [{}])[0].get('description', 'No description').title(),
                    "Icon": f"http://openweathermap.org/img/wn/{day_data.get('weather', [{}])[0].get('icon', '01d')}@2x.png"
                }
                forecast_list.append(forecast)
        
        print(f"[Weather] Successfully extracted {len(forecast_list)} days of forecast for the race weekend.")
        return forecast_list

    except requests.exceptions.HTTPError as http_err:
        print(f"--- WEATHER API HTTP ERROR --- (Status: {http_err.response.status_code})")
        if http_err.response.status_code == 401:
            print("-> This is an Authentication error. Please check that your OpenWeatherMap API key is correct and active.")
        else: print(f"-> Response Body: {http_err.response.text}")
        return []
    except Exception as e:
        print(f"[Weather] General error fetching forecast: {e}")
        return []

@lru_cache(maxsize=32)
def get_championship_standings_progression(year, event_round):
    print(f"[get_championship_standings_progression] Called for Year: {year}, up to Round: {event_round}")
    ergast_api = ergast.Ergast(); print(f"[get_championship_standings_progression] Ergast client initialized.")
    all_driver_standings, all_constructor_standings = [], []
    # Robust check for event_round type and value
    is_valid_round = False
    if event_round is not None:
        try:
            event_round_int_check = int(event_round)
            if event_round_int_check > 0:
                is_valid_round = True
        except (ValueError, TypeError):
            is_valid_round = False
            
    if not is_valid_round: 
        print(f"[get_championship_standings_progression] Invalid or non-positive event_round: {event_round}. Returning empty DFs.")
        return pd.DataFrame(), pd.DataFrame()
    
    event_round_int = int(event_round) # Now safe to convert

    for r in range(1, event_round_int + 1):
        print(f"[get_championship_standings_progression] Attempting to fetch data for round {r}...")
        try:
            driver_st_content = ergast_api.get_driver_standings(season=year, round=r).content
            if driver_st_content: driver_st_df = driver_st_content[0].reset_index(); driver_st_df['round'] = r; all_driver_standings.append(driver_st_df)
            constructor_st_content = ergast_api.get_constructor_standings(season=year, round=r).content
            if constructor_st_content: constructor_st_df = constructor_st_content[0].reset_index(); constructor_st_df['round'] = r; all_constructor_standings.append(constructor_st_df)
        except Exception as e: print(f"[get_championship_standings_progression] Could not fetch/process standings for round {r}, Y{year}: {e}"); continue
    if not all_driver_standings and not all_constructor_standings and event_round_int > 0: print("[get_championship_standings_progression] No standings data fetched."); return pd.DataFrame(), pd.DataFrame()
    driver_df = pd.concat(all_driver_standings, ignore_index=True) if all_driver_standings else pd.DataFrame()
    constructor_df = pd.concat(all_constructor_standings, ignore_index=True) if all_constructor_standings else pd.DataFrame()
    if not driver_df.empty and 'points' in driver_df.columns: driver_df['points'] = pd.to_numeric(driver_df['points'])
    if not constructor_df.empty and 'points' in constructor_df.columns: constructor_df['points'] = pd.to_numeric(constructor_df['points'])
    print(f"[get_championship_standings_progression] Completed. Drivers: {len(driver_df)} pts, Constructors: {len(constructor_df)} pts.")
    return driver_df, constructor_df

def get_race_highlights_from_perplexity(race_summary_data_str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "AI highlights unavailable - API key not configured."
    
    if not race_summary_data_str:
        return "Not enough race data for AI highlights."

    prompt = (
    "You are an F1 race analyst. Based on the following F1 session information, "
    "provide 3-4 key highlights of the race in bullet point format. "
    "Focus on interesting outcomes, significant events implied by the data, "
    "or notable performances. Be concise and use markdown for bullet points. "
    "Do not include citations, reference numbers, or source links in your answer.\n\n"
    f"Session Information:\n{race_summary_data_str}\n\nYour 3-4 bullet point highlights:")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    json_payload = {
        "model": "sonar-pro",   # Example model name, check your Perplexity docs or dashboard for available models
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=json_payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        # Extracting response text from the 'choices' list as in OpenAI-style APIs
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "AI highlights currently unavailable."
    except Exception as e:
        print(f"Perplexity API error for race highlights: {e}")
        return "Could not generate AI highlights due to an API error."



# --- REPLACE your existing get_ai_team_summary function with this one ---
def get_ai_team_summary(team_name, summary_type, context=""):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "AI insights are disabled - API key not configured."

    if summary_type == "history":
        user_prompt = (f"Provide a brief 1-2 sentence history of the Formula 1 team {team_name}, focusing on their origins and key achievements. "
    "Do not include citations, reference numbers, or source links in your answer.")
    elif summary_type == "performance":
        user_prompt = (f"You are an F1 expert analyst. Based on the following data about the {team_name} team performance, "
    "provide a concise 1-2 sentence summary. Do not include citations, reference numbers, or source links in your answer. "f"{context}")
    else:
        return "Information unavailable."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    json_payload = {
        "model": "sonar-pro",  # Adjust if you have access to other models
        "messages": [
            {"role": "system", "content": "You are an F1 data analyst. Provide a concise, factual summary with NO citations, NO reference numbers, "
    "and NO source links—just clear narrative text."
    "\n\n[existing prompt content here]\n\n"},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=json_payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "AI summary currently unavailable."
    except Exception as e:
        print(f"Perplexity API error for {team_name}: {e}")
        return "Could not generate AI summary due to an API error."



@lru_cache(maxsize=2)
def get_all_teams_data(year):
    print(f"[get_all_teams_data] Fetching data for all {year} teams...")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            print(f"Could not load event schedule for {year}.")
            return []

        # Get the latest completed race to fetch standings
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        completed_races = (
            schedule[schedule['EventDate'] < pd.Timestamp.now()]
            .sort_values(by='EventDate', ascending=False)
            .copy()
        )
        if completed_races.empty:
            print("[get_all_teams_data] No completed races found for this year yet.")
            return []

        latest_race = completed_races.iloc[0]
        print(f"Using latest race: {latest_race['EventName']}")  # Debug line

        # Fetch constructor standings from Ergast (handle list case)
        ergast_api = ergast.Ergast()
        constructor_standings_df = pd.DataFrame()
        try:
            round_number = int(latest_race['RoundNumber'])  # Ensure it's an integer
            standings_content = ergast_api.get_constructor_standings(season=year, round=round_number).content
            # Ergast returns a list of DataFrames; get the first if available
            if isinstance(standings_content, list) and len(standings_content) > 0:
                constructor_standings_df = standings_content[0]
            else:
                constructor_standings_df = pd.DataFrame()
        except Exception as e_ergast:
            print(f"Could not fetch constructor standings: {e_ergast}")

        # Get team and driver info from a minimal session load
        session = ff1.get_session(year, latest_race['EventName'], 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        
        if results is None or results.empty:
            print("Could not load session results for the latest race.")
            return []

        teams_data = []
        unique_teams = results['TeamName'].dropna().unique()
        
        for team_name in unique_teams:
            team_drivers = results[results['TeamName'] == team_name]
            driver_list = team_drivers[['Abbreviation', 'FullName']].to_dict('records')
            
            official_constructor_name = TEAM_NAME_TO_CONSTRUCTOR_NAME_MAP.get(team_name, team_name)
            final_color = CONSTRUCTOR_NAME_TO_COLOR_MAP.get(official_constructor_name, "#CCCCCC")
            
            team_standing_info = "Current championship position data is unavailable."
            if not constructor_standings_df.empty:
                team_in_standings = constructor_standings_df[constructor_standings_df['constructorName'] == official_constructor_name]
                if not team_in_standings.empty:
                    standing = team_in_standings.iloc[0]
                    team_standing_info = (
                        f"They are currently P{standing['position']} "
                        f"in the constructors' championship with {standing['points']} points after Round {latest_race['RoundNumber']}."
                    )

            team_info = {
                "TeamName": team_name, "TeamColor": final_color, "Drivers": driver_list,
                "History": get_ai_team_summary(team_name, "history"),
                "Performance": get_ai_team_summary(team_name, "performance", team_standing_info)
            }
            teams_data.append(team_info)
        
        return teams_data
    except Exception as e:
        import traceback
        print(f"Major error in get_all_teams_ {e}")
        traceback.print_exc()
        return []

# --- MAIN DATA FETCHING FUNCTION (get_session_results) --- (MODIFIED for P_OVERVIEW table)
@lru_cache(maxsize=32)
def get_session_results(year, event_name_or_round, session_type='Q'):
    event_round_num = None; podium_data = []; fastest_lap_driver_name_for_table = None
    # Initialize with empty DataFrames
    display_df = pd.DataFrame()
    # raw_results_for_graph will be session.results for Q,R,S, and specific structure for P_OVERVIEW
    raw_results_for_graph = pd.DataFrame() 
    actual_session_name = session_type

    try:
        print(f"[get_session_results] Args: Year={year}, Event='{event_name_or_round}', Session='{session_type}'")

        if session_type == 'P_OVERVIEW':
            actual_session_name = "Practice Overview"
            try:
                temp_event_obj = ff1.get_event(year, event_name_or_round)
                event_round_num = temp_event_obj.get_round_number()
            except Exception as e_event_lookup: 
                print(f"Could not get event details for {year} {event_name_or_round} (Practice Overview): {e_event_lookup}")
            
            # This dictionary will store: DriverAbbr -> {'DriverName', 'TeamName', 'FP1': Timedelta, 'FP2': Timedelta, 'FP3': Timedelta, '_OverallBestRaw': Timedelta}
            driver_fp_data = {} 

            for fp_identifier in ['FP1', 'FP2', 'FP3']:
                try:
                    print(f"[get_session_results] Loading {fp_identifier} for {event_name_or_round}")
                    session_fp = ff1.get_session(year, event_name_or_round, fp_identifier)
                    session_fp.load(laps=True, telemetry=False, weather=False, messages=False)
                    if not session_fp.laps.empty:
                        best_laps_in_this_fp = session_fp.laps.loc[session_fp.laps.groupby('Driver')['LapTime'].idxmin()]
                        for _, row in best_laps_in_this_fp.iterrows():
                            driver_abbr = row['Driver']
                            lap_time = row['LapTime']
                            
                            if driver_abbr not in driver_fp_data:
                                driver_name_full = row.get('DriverFullName', driver_abbr)
                                if pd.isna(driver_name_full) or not driver_name_full: driver_name_full = driver_abbr
                                driver_fp_data[driver_abbr] = {
                                    'Driver': driver_name_full, 
                                    'Team': row['Team'],
                                    'FP1': pd.NaT, 'FP2': pd.NaT, 'FP3': pd.NaT,
                                    '_OverallBestRaw': pd.NaT # For sorting to get position
                                }
                            
                            driver_fp_data[driver_abbr][fp_identifier] = lap_time # Store as Timedelta

                            # Update overall best time for this driver
                            if pd.isna(driver_fp_data[driver_abbr]['_OverallBestRaw']) or \
                               (pd.notna(lap_time) and lap_time < driver_fp_data[driver_abbr]['_OverallBestRaw']):
                                driver_fp_data[driver_abbr]['_OverallBestRaw'] = lap_time
                except Exception as e_fp_load: 
                    print(f"Error loading/processing {fp_identifier} for {event_name_or_round} {year}: {e_fp_load}")
            
            if not driver_fp_data: 
                return pd.DataFrame(), pd.DataFrame(), actual_session_name, event_round_num, podium_data

            # Create display_df for the table
            table_list = []
            for driver_abbr, data in driver_fp_data.items():
                table_list.append({
                    'Driver': data['Driver'],
                    'Team': data['Team'],
                    'FP1 Time': format_timedelta(data['FP1']),
                    'FP2 Time': format_timedelta(data['FP2']),
                    'FP3 Time': format_timedelta(data['FP3']),
                    '_OverallBestRaw': data['_OverallBestRaw'] # Keep for sorting
                })
            display_df = pd.DataFrame(table_list)
            
            if not display_df.empty:
                display_df.sort_values(by='_OverallBestRaw', inplace=True, na_position='last')
                display_df.insert(0, 'Pos', range(1, len(display_df) + 1))
                display_df = display_df[['Pos', 'Driver', 'Team', 'FP1 Time', 'FP2 Time', 'FP3 Time']] # Final columns and order

            # Create raw_results_for_graph (for the line chart - this was working well for you)
            chart_data_list = []
            for driver_abbr, data in driver_fp_data.items(): # Iterate over driver_fp_data which has Timedeltas
                chart_data_list.append({
                    'DriverName': data['Driver'], 
                    'TeamName': data['Team'],
                    'FP1_Time_seconds': data['FP1'].total_seconds() if pd.notna(data['FP1']) else np.nan,
                    'FP2_Time_seconds': data['FP2'].total_seconds() if pd.notna(data['FP2']) else np.nan,
                    'FP3_Time_seconds': data['FP3'].total_seconds() if pd.notna(data['FP3']) else np.nan,
                })
            raw_results_for_graph = pd.DataFrame(chart_data_list)
            
            return display_df, raw_results_for_graph, actual_session_name, event_round_num, podium_data
        
        # --- Existing logic for Q, R, S sessions ---
        # (This should be your already working logic from the "Past Seasons" tab)
        # It correctly returns: display_df, raw_results_for_graph (which is session.results),
        # actual_session_name, event_round_num, podium_data
        # Your previously working code for Q, R, S is assumed to be here.
        session = ff1.get_session(year, event_name_or_round, session_type)
        event_round_num = session.event['RoundNumber']
        load_laps = True if session_type in ['R', 'S'] else False
        session.load(laps=load_laps, telemetry=False, weather=False, messages=False)
        raw_results_for_graph = session.results 
        actual_session_name = session.name
        if raw_results_for_graph is None or raw_results_for_graph.empty: return pd.DataFrame(), pd.DataFrame(), actual_session_name, event_round_num, []
        if 'BroadcastName' in raw_results_for_graph.columns and 'Abbreviation' in raw_results_for_graph.columns: raw_results_for_graph['DriverName'] = raw_results_for_graph.apply(lambda r: r['BroadcastName'] if pd.notna(r['BroadcastName']) and r['BroadcastName'] else r['Abbreviation'], axis=1)
        elif 'Abbreviation' in raw_results_for_graph.columns: raw_results_for_graph['DriverName'] = raw_results_for_graph['Abbreviation']
        else: raw_results_for_graph['DriverName'] = "N/A"
        if session_type in ['R', 'S'] and 'Points' in raw_results_for_graph.columns: raw_results_for_graph['Points'] = pd.to_numeric(raw_results_for_graph['Points'], errors='coerce').fillna(0)
        # Inside get_session_results, for R/S sessions, before podium extraction:
        if session_type in ['R', 'S']:
            print(f"\n[DEBUG get_session_results - Podium Data Check for {year} {event_name_or_round} {session_type}]")
            if not raw_results_for_graph.empty and 'Position' in raw_results_for_graph.columns:
                print(f"Raw results head for podium:\n{raw_results_for_graph[['Position', 'DriverName', 'TeamName', 'Abbreviation', 'TeamColor']].head(5).to_string()}")
        else:
            print("Raw results for podium is empty or missing 'Position' column.")
# ... then the existing podium extraction logic ...

        if session_type in ['R', 'S'] and 'Position' in raw_results_for_graph.columns:
            raw_results_for_graph['Position'] = pd.to_numeric(raw_results_for_graph['Position'], errors='coerce')
            pod_cand = raw_results_for_graph[raw_results_for_graph['Position'].isin([1.0,2.0,3.0])].sort_values(by='Position')
            for _, r_row in pod_cand.iterrows(): 
                tc_val = r_row.get('TeamColor',None); tc_hex = ('#'+tc_val) if tc_val and isinstance(tc_val,str) and not tc_val.startswith('#') else tc_val
                podium_data.append({'Position':int(r_row['Position']),'DriverName':r_row['DriverName'],'TeamName':r_row['TeamName'],'Abbreviation':r_row.get('Abbreviation',''),'TeamColor':tc_hex})
        if session_type == 'Q':
            rel_cols = ['DriverName','TeamName','Position','Q1','Q2','Q3']; ex_cols = [c for c in rel_cols if c in raw_results_for_graph.columns]
            if 'Position' not in ex_cols: return pd.DataFrame(),raw_results_for_graph,actual_session_name,event_round_num,podium_data
            display_df = raw_results_for_graph[ex_cols].copy(); display_df.rename(columns={'DriverName':'Driver','TeamName':'Team','Position':'Pos'},inplace=True)
            for c in ['Q1','Q2','Q3']:
                if c in display_df: display_df[c] = display_df[c].apply(format_timedelta)
            if 'Pos' in display_df.columns: display_df=display_df.sort_values(by='Pos',na_position='last')
        elif session_type in ['R', 'S']:
            rel_cols = ['DriverName','TeamName','Position','Time','Status','Laps','GridPosition','Points']; ex_cols = [c for c in rel_cols if c in raw_results_for_graph.columns]
            # Ensure 'Positions Gained' is calculated if source columns exist, even if not in rel_cols initially
            if 'GridPosition' in raw_results_for_graph.columns and 'Position' in raw_results_for_graph.columns:
                 raw_results_for_graph['Starting Position Temp'] = pd.to_numeric(raw_results_for_graph['GridPosition'], errors='coerce')
                 raw_results_for_graph['Final Position Temp'] = pd.to_numeric(raw_results_for_graph['Position'], errors='coerce')
                 raw_results_for_graph['Positions Gained'] = (raw_results_for_graph['Starting Position Temp'] - raw_results_for_graph['Final Position Temp']).fillna(0).astype(int)
                 if 'Positions Gained' not in rel_cols: rel_cols.append('Positions Gained') # Add if calculated

            ex_cols = [c for c in rel_cols if c in raw_results_for_graph.columns] # Re-evaluate existing_cols
            if not ex_cols: return pd.DataFrame(),raw_results_for_graph,actual_session_name,event_round_num,podium_data
            display_df = raw_results_for_graph[ex_cols].copy(); display_df.rename(columns={'DriverName':'Driver','TeamName':'Team','Position':'Pos','GridPosition':'Starting Position'},inplace=True)
            if 'Time' in display_df.columns: display_df['Time'] = display_df['Time'].apply(format_timedelta)
            if load_laps:
                try:
                    fl_info = session.laps.pick_fastest()
                    if fl_info is not None and pd.notna(fl_info['Driver']):
                        fl_abbr = fl_info['Driver']
                        if 'Abbreviation' in raw_results_for_graph.columns:
                            dn_series = raw_results_for_graph.loc[raw_results_for_graph['Abbreviation']==fl_abbr,'DriverName']
                            if not dn_series.empty: fastest_lap_driver_name_for_table = dn_series.iloc[0]
                except Exception as e: print(f"FL Error: {e}")
            if 'Pos' in display_df.columns: display_df=display_df.sort_values(by='Pos',na_position='last')
        else: # Fallback if session_type is not P_OVERVIEW, Q, R, or S
             return pd.DataFrame(), raw_results_for_graph, f"Unsupported type '{session_type}'", event_round_num, podium_data

        if fastest_lap_driver_name_for_table and 'Driver' in display_df.columns:
            display_df['is_fastest_lap_holder']=(display_df['Driver']==fastest_lap_driver_name_for_table)
        else:
            display_df['is_fastest_lap_holder']=False
            
        return display_df, raw_results_for_graph, actual_session_name, event_round_num, podium_data
    except ff1.ErgastMissingDataError: return pd.DataFrame(), pd.DataFrame(), f"{session_type} (Data Missing)", event_round_num, []
    except Exception as e: print(f"Error in get_session_results for {session_type}: {e}"); return pd.DataFrame(), pd.DataFrame(), f"{session_type} (Error)", event_round_num, []

# --- REPLACE your existing get_next_race_info function with this FINAL version ---
@lru_cache(maxsize=32)
def get_next_race_info(year):
    # This function is now stable and fetches all necessary data, including tyre stints.
    print(f"\n[get_next_race_info] Finding next race for {year}...")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        now_local = pd.Timestamp.now()
        future_races = schedule[schedule['EventDate'] > now_local].sort_values(by='EventDate')
        
        if future_races.empty: return None
        next_event_series = future_races.iloc[0]
        event_name = next_event_series['EventName']
        print(f"[get_next_race_info] Next event found: {event_name}")
        
        data = {
            "EventName": next_event_series['OfficialEventName'], "Country": next_event_series['Country'],
            "CircuitName": next_event_series['Location'], "RaceDate": "TBC", "NumberOfLaps": "TBC",
            "CircuitLength": "TBC", "RaceDistance": "TBC", "LastYearsFastestLap": "N/A",
            "CircuitImageRelativePath": "images/circuit_default.png",
            "CountryFlagImageRelativePath": f"images/flags/{next_event_series['Country'].lower().replace(' ', '_')}.png",
            "SessionSchedule": [], "PastWinners": [], "TyreStrategyData": pd.DataFrame(),
            "WeatherData": [], "DriverOrder": [] 
        }

        race_date_obj = pd.to_datetime(next_event_series['EventDate'])
        if pd.notna(race_date_obj): data['RaceDate'] = race_date_obj.strftime('%d/%m/%y')

        lat, lon, circuit_length_km = None, None, "TBC"
        if not circuit_data_df.empty:
            circuit_details = circuit_data_df[circuit_data_df['RoundNumber'] == next_event_series['RoundNumber']]
            if not circuit_details.empty:
                if 'CircuitLength_km' in circuit_details.columns:
                    circuit_length_km = circuit_details['CircuitLength_km'].iloc[0]
                    data['CircuitLength'] = f"{circuit_length_km} km" if isinstance(circuit_length_km, (int, float)) else "TBC"
                if 'CircuitImageFile' in circuit_details.columns and pd.notna(circuit_details.iloc[0]['CircuitImageFile']):
                    specific_path = f"images/circuits/{circuit_details.iloc[0]['CircuitImageFile']}"
                    if os.path.exists(os.path.join(assets_folder, specific_path)): data['CircuitImageRelativePath'] = specific_path
                if 'Latitude' in circuit_details.columns and 'Longitude' in circuit_details.columns:
                    lat, lon = circuit_details['Latitude'].iloc[0], circuit_details['Longitude'].iloc[0]
        
        if lat and lon and 'race_date_obj' in locals() and pd.notna(race_date_obj):
            data['WeatherData'] = get_weather_forecast(lat, lon, OPENWEATHER_API_KEY, race_date_obj)
        
        try:
            last_year_session = ff1.get_session(year - 1, event_name, 'R'); last_year_session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps_df = last_year_session.laps; results_df = last_year_session.results
            
            if not results_df.empty and 'Laps' in results_df.columns and pd.notna(results_df.iloc[0]['Laps']): data['NumberOfLaps'] = int(results_df.iloc[0]['Laps'])
            elif not laps_df.empty: data['NumberOfLaps'] = int(laps_df['LapNumber'].max())
            if isinstance(data['NumberOfLaps'], int) and isinstance(circuit_length_km, (int, float)):
                data['RaceDistance'] = f"{round(data['NumberOfLaps'] * circuit_length_km, 2)} km"
            
            if not laps_df.empty:
                fastest = laps_df.pick_fastest()
                if fastest is not None and pd.notna(fastest['LapTime']): data['LastYearsFastestLap'] = f"{format_timedelta(fastest['LapTime'])} by {fastest['Driver']} ({year-1})"
                
                # --- Tyre Strategy Data Logic (for go.Bar loop) ---
                if not results_df.empty:
                    driver_order_list = results_df.sort_values(by="Position")['Abbreviation'].unique().tolist()
                    data['DriverOrder'] = driver_order_list
                    stints = laps_df.loc[:, ['Driver', 'Stint', 'Compound', 'LapNumber']]
                    stints = stints.groupby(['Driver', 'Stint', 'Compound']).agg(LapStart=('LapNumber', 'min'), LapEnd=('LapNumber', 'max')).reset_index()
                    data['TyreStrategyData'] = stints

                # --- NEW: Get Safety Car Data ---
            if hasattr(last_year_session, 'race_control_messages') and not last_year_session.race_control_messages.empty:
                rc_messages = last_year_session.race_control_messages
                # Count messages where Category is 'SafetyCar'. Divide by 2 for Deploy/In cycle.
                safety_car_count = len(rc_messages[rc_messages['Message'].str.contains('SAFETY CAR DEPLOYED')])
                data['SafetyCarsLastYear'] = safety_car_count
                print(f"[get_next_race_info] Found {safety_car_count} safety car deployments last year.")

        except Exception as e: print(f"WARNING: Could not load or process last year's session data. Error: {e}")

        # --- Get Session Schedule for This Year ---
        try:
            from pytz import timezone
            utc, pacific = timezone('UTC'), timezone('US/Pacific')
            schedule_items = []
            for i in range(1, 6):
                session_name = next_event_series.get(f'Session{i}')
                session_date_utc = next_event_series.get(f'Session{i}DateUtc')
                if pd.notna(session_name) and pd.notna(session_date_utc):
                    dt_utc = pd.to_datetime(session_date_utc)
                    if dt_utc.tzinfo is None:
                        dt_utc = utc.localize(dt_utc)
                    pacific_time = dt_utc.astimezone(pacific)
                    data['SessionSchedule'].append({'Session': session_name, 'Date': pacific_time.strftime('%a, %b %d'), 'Time': pacific_time.strftime('%I:%M %p PT')})
            print(f"[get_next_race_info] SUCCESS: Found {len(data['SessionSchedule'])} sessions for schedule.")
        except Exception as e:
            print(f"[get_next_race_info] FAILED to process session schedule. Error: {e}")
        
        # --- Get Past 3 Winners ---
        try:
            print("[get_next_race_info] Fetching past winners...")
            for i in range(1, 4):
                past_year = year - i
                try:
                    past_session = ff1.get_session(past_year, event_name, 'R'); past_session.load(laps=True)
                    if not past_session.results.empty:
                        winner_row = past_session.results.loc[past_session.results['Position'] == 1.0].iloc[0]
                        winner_abbr = winner_row['Abbreviation']
                        winner_laps = past_session.laps.pick_drivers([winner_abbr])
                        winner_best_lap = "N/A"
                        if not winner_laps.empty:
                            winner_best_lap = format_timedelta(winner_laps['LapTime'].min())
                        data['PastWinners'].append({'Year': past_year, 'Winner': winner_row['BroadcastName'], 'Team': winner_row['TeamName'], 'BestLap': winner_best_lap, 'Abbreviation': winner_abbr})
                except Exception as e_inner:
                    print(f"Could not get winner for {past_year} {event_name}: {e_inner}")
            print(f"[get_next_race_info] SUCCESS: Found {len(data['PastWinners'])} past winners.")
        except Exception as e:
            print(f"ERROR fetching past winners block: {e}")

        return data

    except Exception as e:
        import traceback
        print(f"FATAL error in get_next_race_info: {e}"); 
        traceback.print_exc()
        return None # CORRECTED: return None, not None()

# --- Initialize the Dash App ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.LUX], 
                suppress_callback_exceptions=True,
                assets_folder='assets' # Explicitly define the assets folder
               )
server = app.server
assets_folder = os.path.join(os.path.dirname(__file__), "assets")

# --- Define the App Layout ---
current_year = datetime.now().year
historical_max_year = current_year - 1
historical_years_options = [{'label': str(y), 'value': y} for y in range(2018, historical_max_year + 1)]
hist_session_type_options = [{'label': 'Qualifying', 'value': 'Q'}, {'label': 'Sprint', 'value': 'S'}, {'label': 'Race', 'value': 'R'}]
cs_session_type_options = [{'label': 'Practice Overview', 'value': 'P_OVERVIEW'}, {'label': 'Qualifying', 'value': 'Q'}, {'label': 'Sprint', 'value': 'S'}, {'label': 'Race', 'value': 'R'}]


# --- Content for Tab 1: Historical Data ---
tab_historical_content = dbc.Card(
    dbc.CardBody([
        dbc.Card([
            dbc.CardHeader(html.H5(f"Filters ({historical_years_options[0]['label']} - {historical_max_year})", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([dbc.Label("Year:"), dcc.Dropdown(id='hist-year-dropdown', options=historical_years_options, value=historical_max_year)], lg=4),
                    dbc.Col([dbc.Label("Event:"), dcc.Dropdown(id='hist-event-dropdown')], lg=4),
                    dbc.Col([dbc.Label("Session:"), dcc.Dropdown(id='hist-session-type-dropdown', options=hist_session_type_options, value='R')], lg=4),
                ]),
            ]),
        ], className="mb-4"),
        html.H4(id='hist-page-title', className="mt-4 mb-3 text-center"),
        dbc.Row(id='hist-podium-display', className="mb-4 justify-content-center text-center g-3"), 
        dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader(html.H5("Race Highlights (AI Generated)",className="mb-0")),dbc.CardBody(dcc.Loading(id="loading-hist-ai-highlights",type="default",children=[dcc.Markdown(id='hist-ai-race-highlights',children="Select a Race...")]))]),width=12)],className="mb-4"),
        dcc.Loading(children=[dash_table.DataTable(id='hist-results-table', style_table={'overflowX': 'auto'}, style_cell={'fontFamily':'Arial','fontSize':'0.9rem','padding':'8px', 'textAlign':'left', 'whiteSpace':'normal', 'height': 'auto'}, style_header={'backgroundColor':'#f8f9fa','fontWeight':'bold', 'borderBottom': '2px solid black'}, page_size=22, style_data_conditional=[])]),
        dcc.Loading(children=[dcc.Graph(id='hist-results-graph')], className="mt-3"),
        html.Div(id='hist-championship-graphs-container', children=[dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id='hist-driver-standings-graph')), md=6), dbc.Col(dcc.Loading(dcc.Graph(id='hist-constructor-standings-graph')), md=6)], className="mt-4")])
    ]), className="mt-3"
)

# --- Content for Tab 2: Current Season (2025) - REORDERED ---
cs_sub_nav_bar = dbc.Nav(
    [
        dbc.NavLink("Season So Far", active=True, href="#cs-season-so-far", id="cs-navlink-season-so-far"), # active=True initially
        dbc.NavLink("Next Race", href="#cs-next-race", id="cs-navlink-next-race", disabled=False), # ENABLED
        dbc.NavLink("Race Calendar", href="#cs-race-calendar", id="cs-navlink-race-calendar", disabled=False), # ENABLED
    ],
    pills=True,
    id="cs-sub-nav", 
    className="mb-3 justify-content-center"
)

season_so_far_content_layout = dbc.Container([ # This is the content for the "Season So Far" sub-page
    html.H5("Session Details & Analysis", className="mt-3 mb-3 text-center", id="cs-page-title"),
    dbc.Row(id='cs-podium-display', className="mb-4 justify-content-center text-center g-3"),
    dbc.Card([dbc.CardHeader(html.H6("Session Highlights (AI Generated)",className="mb-0")),
              dbc.CardBody(dcc.Loading(dcc.Markdown(id='cs-ai-highlights',children="Select a session...")))], className="mb-4"),
    dcc.Loading(id="cs-loading-table", children=[
        dash_table.DataTable(id='cs-results-table', style_table={'overflowX': 'auto'}, 
                             style_cell={'fontFamily':'Arial','fontSize':'0.9rem','padding':'8px'}, 
                             style_header={'backgroundColor':'#f8f9fa','fontWeight':'bold'}, 
                             page_size=22, style_data_conditional=[])]),
    dcc.Loading(id="cs-loading-graph", children=[dcc.Graph(id='cs-results-graph')], className="mt-3"),
    html.Div(id='cs-championship-graphs-container', children=[
        dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id='cs-driver-standings-graph')), md=6), 
                 dbc.Col(dcc.Loading(dcc.Graph(id='cs-constructor-standings-graph')), md=6)], className="mt-4")])
], fluid=True)

next_race_content_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H4(id="cs-next-race-title", className="text-center fw-bold"))),
    html.Hr(),
    dbc.Row([
        # Left Column: Track Map
        dbc.Col(
            html.Div(
                html.Img(id="cs-circuit-map-img", className="img-fluid") # The track layout image will go here
            ), 
            lg=7, md=12, className="mb-3"
        ),
        # Right Column: Key Stats
        dbc.Col(
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div([html.P("First Grand Prix", className="text-muted small mb-0"), html.H4(id="cs-first-gp", className="fw-bold")])),
                    dbc.Col(html.Div([html.P("Number of Laps", className="text-muted small mb-0"), html.H4(id="cs-laps", className="fw-bold")])),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(html.Div([html.P("Circuit Length", className="text-muted small mb-0"), html.H4(id="cs-circuit-length", className="fw-bold")])),
                    dbc.Col(html.Div([html.P("Race Distance", className="text-muted small mb-0"), html.H4(id="cs-race-distance", className="fw-bold")])),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.P("Lap Record", className="text-muted small mb-0"), 
                            html.H4(id="cs-lap-record-time", className="fw-bold d-inline-block"),
                            html.Span(id="cs-lap-record-holder", className="ms-2 text-muted")
                        ]),
                        width=12
                    )
                ], className="mb-4")
                # We can add the other links (Onboard Lap, etc.) later if you wish
            ]), 
            lg=5, md=12
        ),
    ], align="center")
], fluid=True, id="next-race-container")

race_calendar_content_layout = dbc.Container([
    html.H5("2025 Race Calendar", className="mt-3 mb-3 text-center"),
    dbc.Alert("The full 2025 race calendar will be displayed here.", color="info"),
    html.Div(id="cs-race-calendar-table-div") # For the calendar table
], fluid=True, id="race-calendar-container") # Add ID for clarity

tab_2025_content = dbc.Card(
    dbc.CardBody([
        html.H4(f"Current Season Insights ({current_year})", className="mb-3 text-center"),
        cs_sub_nav_bar, # This is your dbc.Nav for sub-tabs
        dbc.Card([ # THIS IS THE FILTERS CARD
            dbc.CardHeader(html.H5("Select Event & Session for " + str(current_year), className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([dbc.Label("Event:"), dcc.Dropdown(id='cs-event-dropdown')], lg=6, md=6),
                    dbc.Col([dbc.Label("Session:"), dcc.Dropdown(id='cs-session-type-dropdown', options=cs_session_type_options, value='R')], lg=6, md=6),
                ]),
            ]),
        ], className="mb-4", id='cs-filters-card'), # <<< ENSURE THIS ID IS PRESENT
        html.Div(id="cs-sub-tab-content-area") 
    ]), className="mt-3"
)

tab_teams_drivers_content = dbc.Card(
    dbc.CardBody([
        html.H4("2025 Teams & Drivers", className="text-center mb-4"),
        # This Div will be populated with all 10 team cards by our new callback
        dcc.Loading(
            id="loading-teams-content",
            type="default",
            children=[html.Div(id="teams-drivers-content-area")]
        )
    ])
)

tab_predictions_content = dbc.Card(
    dbc.CardBody([
        html.H4("Machine Learning Predictions", className="text-center mb-2"),
        dbc.Alert(
            "This page simulates the model retraining process for each completed race of the 2025 season. As it's computationally intensive, it may take a minute to load for the first time.",
            color="info"
        ),
        html.Hr(),
        # This Div will be populated by our new callback
        dcc.Loading(
            id="loading-predictions-content",
            type="default",
            children=[html.Div(id="predictions-content-area")]
        ),
        html.Hr(className="my-4"),
        # Section for predicting the next race
        dbc.Card([
            dbc.CardHeader(html.H5("Predict Next Race")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='predict-race-dropdown', placeholder="Select an upcoming race..."), md=8),
                    dbc.Col(dbc.Button("Generate Prediction", id='predict-race-button', n_clicks=0, color="primary"), md=4),
                ], align="center"),
                dcc.Loading(html.Div(id="next-race-prediction-output", className="mt-3"))
            ])
        ])
    ])
)

# --- Placeholder content for other tabs ---

tab_fantasy_creator_content = dbc.Card(dbc.CardBody([html.P("Content for Fantasy Team Creator tool will be built here.")]), className="mt-3")

# --- START: NEW FANTASY RULES LAYOUT ---

# Data for the scoring tables, based on the F1 Fantasy rules PDF
qualifying_sprint_points_data = [
    {'position': 'P1', 'qualifying': 10, 'sprint': 8}, {'position': 'P2', 'qualifying': 9, 'sprint': 7},
    {'position': 'P3', 'qualifying': 8, 'sprint': 6}, {'position': 'P4', 'qualifying': 7, 'sprint': 5},
    {'position': 'P5', 'qualifying': 6, 'sprint': 4}, {'position': 'P6', 'qualifying': 5, 'sprint': 3},
    {'position': 'P7', 'qualifying': 4, 'sprint': 2}, {'position': 'P8', 'qualifying': 3, 'sprint': 1},
    {'position': 'P9', 'qualifying': 2, 'sprint': 0}, {'position': 'P10', 'qualifying': 1, 'sprint': 0},
]

grand_prix_points_data = [
    {'position': 'P1', 'points': 25}, {'position': 'P2', 'points': 18},
    {'position': 'P3', 'points': 15}, {'position': 'P4', 'points': 12},
    {'position': 'P5', 'points': 10}, {'position': 'P6', 'points': 8},
    {'position': 'P7', 'points': 6}, {'position': 'P8', 'points': 4},
    {'position': 'P9', 'points': 2}, {'position': 'P10', 'points': 1},
]

other_driver_points_data = [
    {'action': 'Overtake Made', 'points': '+1'},
    {'action': 'Fastest Lap', 'points': '+5'},
    {'action': 'Driver of the Day', 'points': '+5'},
    {'action': 'Not Classified (DNF)', 'points': '-15'},
    {'action': 'Disqualified', 'points': '-20'},
]

constructor_points_data = [
    {'action': 'Fastest Pit Stop', 'points': '+5'},
    {'action': 'Fastest Pit Stop (World Record)', 'points': '+15 (Bonus)'},
    {'action': 'Disqualified (per driver)', 'points': '-20'},
]

# This is the main layout variable for the fantasy rules tab
tab_fantasy_rules_content = dbc.Card(
    dbc.CardBody([
        dbc.Container([
            dbc.Row(dbc.Col(html.H2("Official F1 Fantasy Rules (2025)"), className="mb-4 text-center")),
            
            # --- Team & Budget ---
            dbc.Card([
                dbc.CardHeader(html.H4("Team Setup & Budget")),
                dbc.CardBody([
                    html.P("Build your dream team with a set budget to score points based on real-life race results."),
                    html.Ul([
                        html.Li(["Your team must consist of ", html.B("5 Drivers"), " and ", html.B("2 Constructors"),"."]),
                        html.Li(["You have an initial budget of ", html.B("$100.0M"), " to spend."]),
                        html.Li("You can make free transfers between race weeks. Additional transfers cost points."),
                    ])
                ])
            ], className="mb-4"),

            # --- Scoring System ---
            dbc.Card([
                dbc.CardHeader(html.H4("Scoring System")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Qualifying & Sprint Points (Drivers)"),
                            dbc.Table.from_dataframe(pd.DataFrame(qualifying_sprint_points_data), striped=True, bordered=True, hover=True)
                        ], md=6),
                        dbc.Col([
                            html.H5("Grand Prix Finishing Points (Drivers)"),
                            dbc.Table.from_dataframe(pd.DataFrame(grand_prix_points_data), striped=True, bordered=True, hover=True)
                        ], md=6),
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Other Driver Points"),
                            dbc.Table.from_dataframe(pd.DataFrame(other_driver_points_data), striped=True, bordered=True, hover=True)
                        ], md=6),
                        dbc.Col([
                            html.H5("Constructor-Specific Points"),
                            dbc.Table.from_dataframe(pd.DataFrame(constructor_points_data), striped=True, bordered=True, hover=True),
                            html.P("Constructors also score the combined total of their drivers' points from all sessions.", className="mt-2 fst-italic")
                        ], md=6),
                    ])
                ])
            ], className="mb-4"),

            # --- Chips & Power-ups ---
            dbc.Card([
                dbc.CardHeader(html.H4("Chips (Power-ups)")),
                dbc.CardBody([
                    html.P("Use chips to gain a strategic advantage. Each can only be used once per season."),
                    html.Ul([
                        html.Li([html.B("Wildcard:"), " Make unlimited free transfers for one race week."]),
                        html.Li([html.B("Triple Captain:"), " The selected driver's score for the race week is tripled."]),
                        html.Li([html.B("Final Fix:"), " Make one driver substitution between qualifying and the race."]),
                    ])
                ])
            ], className="mb-4"),
            
            # --- Team Budget and Values ---
            dbc.Card([
                dbc.CardHeader(html.H4("Team Budget and Values")),
                dbc.CardBody([
                    html.P([
                        "Total team budget: ", html.B("$113.7M"), 
                        ". Current driver and constructor values are shown below."
                    ]),
                    
                    dbc.Row([
                        # Drivers Table
                        dbc.Col([
                            html.H5("Driver Values", className="mb-3"),
                            dbc.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("", style={"width": "50px"}),  # Image column
                                        html.Th("Driver"),
                                        html.Th("Value")
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/nor.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Lando Norris"),
                                        html.Td("$30.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/pia.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Oscar Piastri"),
                                        html.Td("$26.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/ver.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Max Verstappen"),
                                        html.Td("$28.5M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/rus.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("George Russell"),
                                        html.Td("$22.7M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/ham.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Lewis Hamilton"),
                                        html.Td("$22.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/lec.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Charles Leclerc"),
                                        html.Td("$23.0M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/ant.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Kimi Antonelli"),
                                        html.Td("$14.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/hul.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Nico Hulkenberg"),
                                        html.Td("$8.6M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/alb.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Alexander Albon"),
                                        html.Td("$13.0M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/str.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Lance Stroll"),
                                        html.Td("$9.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/bea.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Oliver Bearman"),
                                        html.Td("$7.3M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/oco.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Esteban Ocon"),
                                        html.Td("$8.3M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/tsu.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Yuki Tsunoda"),
                                        html.Td("$9.8M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/had.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Isack Hadjar"),
                                        html.Td("$6.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/sai.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Carlos Sainz"),
                                        html.Td("$5.7M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/bor.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Gabriel Bortoleto"),
                                        html.Td("$5.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/law.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Liam Lawson"),
                                        html.Td("$5.9M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/alo.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Fernando Alonso"),
                                        html.Td("$7.3M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/gas.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Pierre Gasly"),
                                        html.Td("$4.5M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/drivers/col.png"), style={"width": "40px", "height": "40px", "border-radius": "50%"})),
                                        html.Td("Franco Colapinto"),
                                        html.Td("$4.5M")
                                    ])
                                ])
                            ], striped=True, hover=True, size="sm")
                        ], md=6),
                        
                        # Constructors Table
                        dbc.Col([
                            html.H5("Constructor Values", className="mb-3"),
                            dbc.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("", style={"width": "50px"}),  # Logo column
                                        html.Th("Constructor"),
                                        html.Th("Value")
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/mclaren.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("McLaren"),
                                        html.Td("$35.1M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/ferrari.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Ferrari"),
                                        html.Td("$30.6M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/red_bull_racing.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Red Bull Racing"),
                                        html.Td("$29.3M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/mercedes.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Mercedes"),
                                        html.Td("$26.6M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/alpine.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Alpine"),
                                        html.Td("$8.1M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/haas_f1_team.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Haas"),
                                        html.Td("$13.2M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/racing_bulls.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Racing Bulls"),
                                        html.Td("$13.0M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/kick_sauber.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Kick Sauber"),
                                        html.Td("$10.7M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/aston_martin.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Aston Martin"),
                                        html.Td("$12.3M")
                                    ]),
                                    html.Tr([
                                        html.Td(html.Img(src=app.get_asset_url("images/teams/williams.png"), style={"width": "40px", "height": "30px"})),
                                        html.Td("Williams"),
                                        html.Td("$17.3M")
                                    ])
                                ])
                            ], striped=True, hover=True, size="sm")
                        ], md=6)
                    ])
                ])
            ], className="mb-4")

        
        ], fluid=True, className="mt-4")
    ])
)
# --- END: NEW FANTASY RULES LAYOUT ---


# --- Main App Layout ---
app.layout = dbc.Container([
    dcc.Store(id='cs-active-selection-store'),
    dbc.Row(dbc.Col(html.H1("F1 Insights & Fantasy Predictor", className="page-main-title"), width=12), className="mb-3 mt-3 text-center"),
    dbc.Tabs([
        dbc.Tab(tab_historical_content, label="Past Seasons", tab_id="tab-historical"),
        dbc.Tab(tab_2025_content, label=f"Current Season ({current_year})", tab_id="tab-current-season"),
        dbc.Tab(tab_teams_drivers_content, label="Teams & Drivers", tab_id="tab-teams-drivers"),
        dbc.Tab(tab_predictions_content, label="Predictions", tab_id="tab-predictions"),
        dbc.Tab(tab_fantasy_rules_content, label="Fantasy Rules", tab_id="tab-fantasy-rules"),
        dbc.Tab(tab_fantasy_creator_content, label="Fantasy Team Creator", tab_id="tab-fantasy-creator"),
    ], id="app-main-tabs", active_tab="tab-current-season"),
    html.Hr(className="my-4"),
    dbc.Row(dbc.Col(html.A("Data sourced using FastF1", href="https://theoehrly.github.io/Fast-F1/", target="_blank"), className="text-center text-muted mb-4"))
], fluid=True, id="app-main-container")

# --- Callbacks for Historical Data Tab ---
@app.callback(
    Output('hist-event-dropdown', 'options'), Output('hist-event-dropdown', 'value'),
    Input('hist-year-dropdown', 'value')
)
def update_hist_event_dropdown(selected_year):
    if selected_year is None: return [], None
    try:
        schedule = ff1.get_event_schedule(selected_year, include_testing=False)
        if schedule.empty: return [], None
        unique_events_df = schedule.drop_duplicates(subset=['EventName']).sort_values(by='RoundNumber')
        event_options = [{'label': f"{row['EventName']} (R{row['RoundNumber']})", 'value': row['EventName']} for idx, row in unique_events_df.iterrows()]
        return event_options, (event_options[0]['value'] if event_options else None)
    except Exception as e: print(f"Error in hist_event_dropdown: {e}"); return [], None

@app.callback(
    Output('hist-results-table', 'data'), Output('hist-results-table', 'columns'),
    Output('hist-results-table', 'style_data_conditional'),
    Output('hist-page-title', 'children'), 
    Output('hist-podium-display', 'children'),
    Output('hist-ai-race-highlights', 'children'), 
    Output('hist-results-graph', 'figure'),
    Output('hist-driver-standings-graph', 'figure'),
    Output('hist-constructor-standings-graph', 'figure'),
    Input('hist-year-dropdown', 'value'), Input('hist-event-dropdown', 'value'),
    Input('hist-session-type-dropdown', 'value')
)
def update_hist_main_content(selected_year, selected_event, selected_session_type):
    if not all([selected_year, selected_event, selected_session_type]):
        return [], [], [], "Select Year, Event, and Session", [], "AI Highlights will appear after selecting a race.", {}, {}, {}
    display_df, raw_results_df, actual_session_name, event_round, podium_data_list = get_session_results(selected_year, selected_event, selected_session_type)
    page_title_text = f"{actual_session_name if actual_session_name else selected_session_type} - {selected_event} ({selected_year})"
    table_data, table_columns, style_data_conditional_fl = [], [], []
    ai_highlights_hist_md = "Select a Race session to view AI-generated highlights." 
    if not display_df.empty:
        df_for_table = display_df.copy() 
        if 'is_fastest_lap_holder' in df_for_table.columns:
            for i, is_fl_holder in enumerate(df_for_table['is_fastest_lap_holder']):
                if is_fl_holder: style_data_conditional_fl.append({'if': {'row_index': i, 'column_id': 'Driver'},'backgroundColor': 'lightgoldenrodyellow', 'fontWeight': 'bold'})
            table_columns_df = df_for_table.drop(columns=['is_fastest_lap_holder'], errors='ignore')
        else: table_columns_df = df_for_table
        table_columns = [{"name": i, "id": i} for i in table_columns_df.columns]; table_data = table_columns_df.to_dict('records')
    podium_display_content = []
    if selected_session_type in ['R', 'S'] and podium_data_list:
        podium_cards = []; position_map = {1: "1st Place", 2: "2nd Place", 3: "3rd Place"}
        default_driver_img_rel_path = "images/driver_default.png"; default_team_logo_rel_path = "images/team_default.png"
        default_driver_src = app.get_asset_url(default_driver_img_rel_path); default_team_src = app.get_asset_url(default_team_logo_rel_path)
        for driver_info in podium_data_list:
            driver_abbr = driver_info.get('Abbreviation', '').lower(); team_name_simple = driver_info.get('TeamName', 'default').lower().replace(' ', '_').replace('-', '_')
            specific_driver_img_rel_path = f"images/drivers/{driver_abbr}.png"; specific_team_logo_rel_path = f"images/teams/{team_name_simple}.png"
            driver_image_src = default_driver_src
            if driver_abbr and os.path.exists(os.path.join(assets_folder, specific_driver_img_rel_path)): driver_image_src = app.get_asset_url(specific_driver_img_rel_path)
            team_logo_src = default_team_src
            if team_name_simple != 'default' and os.path.exists(os.path.join(assets_folder, specific_team_logo_rel_path)): team_logo_src = app.get_asset_url(specific_team_logo_rel_path)
            card_style = {}; 
            if driver_info.get('TeamColor') and pd.notna(driver_info['TeamColor']): card_style = {'borderLeft': f"5px solid {driver_info['TeamColor']}"}
            podium_cards.append(dbc.Col(dbc.Card([html.Img(src=driver_image_src, className="img-fluid rounded-circle p-2 mx-auto d-block", style={'width': '100px', 'height': '100px', 'objectFit': 'cover'}),
                    dbc.CardBody([html.H5(position_map.get(driver_info['Position'], f"Pos {driver_info['Position']}"), className="card-title"),
                        html.H6(driver_info['DriverName'], className="card-subtitle mb-2"),
                        html.Img(src=team_logo_src, style={'height': '25px', 'marginRight': '5px', 'verticalAlign': 'middle'}),
                        html.Span(driver_info['TeamName'], className="card-text text-muted small")])
                ], className="text-center h-100 shadow-sm", style=card_style), lg=3, md=4, sm=6, className="mb-3"))
        if podium_cards: podium_display_content = dbc.Row(podium_cards, justify="center")
    if selected_session_type == 'R' and not raw_results_df.empty:
        summary_parts_hist = [f"Race: {selected_event} ({selected_year})"]
        if podium_data_list: summary_parts_hist.append("Podium:"); [summary_parts_hist.append(f"  {p['Position']}. {p['DriverName']} ({p['TeamName']})") for p in podium_data_list]
        pole_sitter_df_hist = raw_results_df[raw_results_df['GridPosition'] == 1]
        if not pole_sitter_df_hist.empty: pole_sitter_hist = pole_sitter_df_hist.iloc[0]; summary_parts_hist.append(f"Pole Sitter: {pole_sitter_hist['DriverName']} ({pole_sitter_hist['TeamName']}), finished P{int(pole_sitter_hist['Position']) if pd.notna(pole_sitter_hist['Position']) else 'N/A'}.")
        fl_driver_row_hist = display_df[display_df['is_fastest_lap_holder'] == True] if 'is_fastest_lap_holder' in display_df else pd.DataFrame()
        if not fl_driver_row_hist.empty: summary_parts_hist.append(f"Fastest Lap: {fl_driver_row_hist.iloc[0]['Driver']}.")
        dnfs_hist = raw_results_df[ (raw_results_df['Status'] != 'Finished') & (~raw_results_df['Status'].str.contains(r'\+', na=False)) & (raw_results_df['GridPosition'] <= 10) ]
        if not dnfs_hist.empty: summary_parts_hist.append("Notable DNFs (from top 10 grid):"); [summary_parts_hist.append(f"  - {dnf_row['DriverName']} ({dnf_row['TeamName']}), Status: {dnf_row['Status']}") for _, dnf_row in dnfs_hist.head(2).iterrows()]
        race_data_summary_hist_str = "\n".join(summary_parts_hist); ai_highlights_hist_md = get_race_highlights_from_perplexity(race_data_summary_hist_str)
    elif selected_session_type != 'R': ai_highlights_hist_md = "AI Race highlights are available for Race sessions only on this tab."
    fig = {} 
    team_color_map = {}
    if not raw_results_df.empty and 'TeamName' in raw_results_df.columns:
        unique_teams = raw_results_df['TeamName'].dropna().unique();
        for team in unique_teams:
            color = None; 
            try: color = ff1_plt.get_team_color(team)
            except Exception : pass
            if color is None and 'TeamColor' in raw_results_df.columns:
                s = raw_results_df.loc[raw_results_df['TeamName'] == team, 'TeamColor']
                if not s.empty and pd.notna(s.iloc[0]): cv = str(s.iloc[0]); color = ('#'+cv) if not cv.startswith('#') else cv
            team_color_map[team] = color if (color and isinstance(color, str) and color.lower() != "nan") else "#CCCCCC"
    if raw_results_df.empty: fig = {'layout': {'title': f'No data for main graph'}}
    elif selected_session_type == 'Q':
        try:
            if 'DriverName' not in raw_results_df.columns: raise ValueError("DriverName missing.")
            id_vars_for_melt = ['DriverName']; value_vars_seconds = []
            if 'TeamName' in raw_results_df.columns: id_vars_for_melt.append('TeamName')
            q_cols = [q for q in ['Q1','Q2','Q3'] if q in raw_results_df.columns]
            if not q_cols: raise ValueError("No Q session columns.")
            q_df = raw_results_df[id_vars_for_melt + q_cols].copy()
            for q_s in q_cols:
                col_sec = q_s + '_seconds'
                if pd.api.types.is_timedelta64_dtype(q_df[q_s]): q_df[col_sec] = q_df[q_s].dt.total_seconds()
                else: q_df[col_sec] = np.nan
                value_vars_seconds.append(col_sec)
            if not value_vars_seconds: raise ValueError("No Q_seconds columns.")
            m_df = q_df.melt(id_vars=id_vars_for_melt, value_vars=value_vars_seconds, var_name='Stage', value_name='Time')
            m_df['Stage'] = m_df['Stage'].str.replace('_seconds',''); m_df.dropna(subset=['Time'], inplace=True)
            if not m_df.empty: fig = px.line(m_df, x='Stage', y='Time', color='DriverName', markers=True, title="Qualifying Times", color_discrete_map=team_color_map if 'TeamName' in id_vars_for_melt else None); fig.update_xaxes(categoryorder='array',categoryarray=['Q1','Q2','Q3']);fig.update_layout(legend_title_text='Driver')
            else: fig = {'layout': {'title': 'No valid quali times'}}
        except Exception as e: print(f"Q graph error: {e}"); fig = {'layout': {'title': f'Q graph error: {e}'}}
    elif selected_session_type in ['R', 'S']:
        if 'Points' in raw_results_df.columns and 'DriverName' in raw_results_df.columns:
            graph_df = raw_results_df[['DriverName', 'TeamName', 'Points']].copy()
            graph_df['Points'] = pd.to_numeric(graph_df['Points'], errors='coerce').fillna(0); graph_df = graph_df[graph_df['Points'] > 0]
            if not graph_df.empty: fig = px.bar(graph_df, x='DriverName', y='Points', title=f"Points Scored", color='TeamName', color_discrete_map=team_color_map); fig.update_layout(xaxis_title="Driver", yaxis_title="Points", legend_title_text='Team', xaxis={'categoryorder':'total descending'})
            else: fig = {'layout': {'title': f'No points scored'}}
        else: fig = {'layout': {'title': f'Points data not available'}}
    else: fig = {'layout': {'title': f'Graph not applicable'}}
    driver_standings_fig, constructor_standings_fig = {}, {}
    if selected_session_type in ['R', 'S'] and isinstance(event_round, (int, np.integer)) and event_round > 0 :
        driver_progression_df, constructor_progression_df = get_championship_standings_progression(selected_year, event_round)
        if not driver_progression_df.empty: driver_standings_fig = px.line(driver_progression_df, x='round', y='points', color='driverCode', title=f"Driver Championship ({selected_year})", labels={'round': 'Rd', 'points': 'Pts', 'driverCode': 'Driver'})
        else: driver_standings_fig = {'layout': {'title': f'Driver Standings: No data'}}
        if not constructor_progression_df.empty:
            constructor_names = constructor_progression_df['constructorName'].unique()
            champ_constructor_color_map = {}
            if constructor_names.size > 0:
                for name in constructor_names:
                    color = team_color_map.get(name) 
                    if not color or color == "#CCCCCC":
                        identifier_to_try = ERGAS_CONSTRUCTOR_NAME_TO_KNOWN_TEAM_ID_OR_COLOR.get(name, name)
                        if isinstance(identifier_to_try, str) and identifier_to_try.startswith("#"): color = identifier_to_try
                        else:
                            try: 
                                specific_constructor_color = ff1_plt.get_team_color(identifier_to_try)
                                if specific_constructor_color: color = specific_constructor_color
                            except TypeError as te:
                                if "missing 1 required positional argument: 'session'" in str(te).lower(): print(f"ChampColor TypeError(session) for '{name}' (mapped to '{identifier_to_try}')")
                                else: print(f"ChampColor TypeError for '{name}' (mapped to '{identifier_to_try}'): {te}")
                            except Exception as e_champ_color: print(f"ChampColor Error for '{name}' (mapped to '{identifier_to_try}'): {e_champ_color}")
                    champ_constructor_color_map[name] = color or "#CCCCCC"
            constructor_standings_fig = px.line(constructor_progression_df, x='round', y='points', color='constructorName', title=f"Constructor Championship ({selected_year})", labels={'round': 'Rd', 'points': 'Pts', 'constructorName': 'Constructor'}, color_discrete_map=champ_constructor_color_map)
        else: constructor_standings_fig = {'layout': {'title': f'Constructor Standings: No data'}}
    else:
        driver_standings_fig = {'layout': {'title': f'Championship graphs NA'}}; constructor_standings_fig = {'layout': {'title': f'Championship graphs NA'}}
    return table_data, table_columns, style_data_conditional_fl, page_title_text, podium_display_content, ai_highlights_hist_md, fig, driver_standings_fig, constructor_standings_fig

# --- Callbacks for Current Season (2025) Tab ---

@app.callback(
    Output('cs-active-selection-store', 'data'),
    Input('cs-event-dropdown', 'value'),
    Input('cs-session-type-dropdown', 'value'),
    State('app-main-tabs', 'active_tab')
)
def store_active_cs_selection(selected_event, selected_session_type, active_tab):
    # This callback updates the store ONLY when the tab is active AND both inputs are valid.
    if active_tab != 'tab-current-season' or not all([selected_event, selected_session_type]):
        # If inputs are not ready, don't change the store's current state
        return dash.no_update 
    
    print(f"[store_active_cs_selection] Storing selection: Event='{selected_event}', Session='{selected_session_type}'")
    return {'event': selected_event, 'session': selected_session_type}

@app.callback(
    Output('cs-event-dropdown', 'options'),
    Output('cs-event-dropdown', 'value'),
    Input('app-main-tabs', 'active_tab') 
)
def update_cs_event_dropdown(active_main_tab):
    # This callback now runs on initial load because its tab is active by default
    triggered_id = dash.callback_context.triggered_id
    
    # We want this to fire on initial load (triggered_id is None) if CS tab is default
    if triggered_id is not None and active_main_tab != "tab-current-season":
        return dash.no_update, dash.no_update

    cs_year = datetime.now().year 
    print(f"[Callback cs_event_dropdown] Populating for Current Season ({cs_year})")
    
    try:
        full_schedule = ff1.get_event_schedule(cs_year, include_testing=False)
        if full_schedule.empty: 
            print(f"[Callback cs_event_dropdown] No events found in schedule for {cs_year}.")
            return [], None
        
        full_schedule['EventDate'] = pd.to_datetime(full_schedule['EventDate'])
        now_local = pd.Timestamp.now() # Naive local time for comparison

        # Filter for completed events
        completed_events_schedule = full_schedule[full_schedule['EventDate'] < now_local].copy()
        
        options_df = pd.DataFrame()
        default_event = None
        
        if not completed_events_schedule.empty:
            # If there are completed events, they are the options
            print(f"[Callback cs_event_dropdown] Found {len(completed_events_schedule)} completed events for {cs_year}.")
            options_df = completed_events_schedule.sort_values(by='RoundNumber').drop_duplicates(subset=['EventName'], keep='first')
            # Default to the LATEST completed event
            default_event = completed_events_schedule.sort_values(by='RoundNumber', ascending=False).iloc[0]['EventName']
        else:
            # If no events completed yet, show all scheduled events
            print(f"[Callback cs_event_dropdown] No completed events yet for {cs_year}. Showing full schedule.")
            options_df = full_schedule.sort_values(by='RoundNumber').drop_duplicates(subset=['EventName'], keep='first')
            # Default to the FIRST event of the season
            if not options_df.empty:
                default_event = options_df.iloc[0]['EventName']

        if options_df.empty:
            return [], None
            
        cs_event_options = [{'label': f"{row['EventName']} (R{row['RoundNumber']})", 'value': row['EventName']} 
                             for idx, row in options_df.iterrows()]
            
        print(f"[Callback cs_event_dropdown] Options generated: {len(cs_event_options)}. Default event: {default_event}")
        return cs_event_options, default_event
    except Exception as e:
        print(f"Error in cs_event_dropdown: {e}")
        return [], None

# Main callback for "Season So Far" sub-section content - NOW TRIGGERED BY THE STORE
@app.callback(
    Output('cs-page-title', 'children'),
    Output('cs-podium-display', 'children'),
    Output('cs-ai-highlights', 'children'),
    Output('cs-results-table', 'data'),
    Output('cs-results-table', 'columns'),
    Output('cs-results-table', 'style_data_conditional'),
    Output('cs-results-graph', 'figure'),
    Output('cs-driver-standings-graph', 'figure'),
    Output('cs-constructor-standings-graph', 'figure'),
    Input('cs-active-selection-store', 'data'), # <<< INPUT IS NOW THE STORE
    State('app-main-tabs', 'active_tab') 
)
def update_cs_season_so_far_content(stored_selection, active_main_tab):
    print(f"[CS_SO_FAR_CONTENT_CALLBACK] Triggered by store. Data: {stored_selection}")

    # Check if the tab is active and if the store has valid data
    if active_main_tab != "tab-current-season" or not stored_selection or \
       not stored_selection.get('event') or not stored_selection.get('session'):
        print("[CS_SO_FAR_CONTENT_CALLBACK] Exiting: No valid selection in store or wrong tab active.")
        return "Current Season Overview", [], "Select an event and session to begin.", [], [], [], {}, {}, {}

    selected_event = stored_selection['event']
    selected_session_type = stored_selection['session']
    
    cs_year = datetime.now().year
    print(f"[CS_SO_FAR_CONTENT_CALLBACK] Processing: Y{cs_year}, Event='{selected_event}', Session='{selected_session_type}'")

    # The rest of your logic from here is IDENTICAL to your last working version.
    # It takes selected_event and selected_session_type and populates all the components.
    
    # ... (Your entire existing logic from calling get_session_results to the final return statement)
    # Ensure it returns all 9 outputs.
    # ...
    display_df, raw_results_for_graph, actual_session_name, event_round, podium_data_list = get_session_results(cs_year, selected_event, selected_session_type)
    page_title_text = f"{actual_session_name if actual_session_name else selected_session_type} - {selected_event} ({cs_year})"
    table_data, table_columns, style_data_conditional_cs, ai_highlights_md, podium_display_cs = [], [], [], "AI Highlights.", []
    fig_cs, driver_standings_fig_cs, constructor_standings_fig_cs = {}, {}, {}
    if not display_df.empty:
        df_for_table = display_df.copy()
        if 'is_fastest_lap_holder' in df_for_table.columns:
            for i, is_fl_holder in enumerate(df_for_table['is_fastest_lap_holder']):
                if is_fl_holder: style_data_conditional_cs.append({'if': {'row_index': i, 'column_id': 'Driver'}, 'backgroundColor': 'lightgoldenrodyellow', 'fontWeight': 'bold'})
            table_columns_df = df_for_table.drop(columns=['is_fastest_lap_holder'], errors='ignore')
        else: table_columns_df = df_for_table
        table_columns = [{"name": col, "id": col} for col in table_columns_df.columns]; table_data = table_columns_df.to_dict('records')
    if selected_session_type in ['R', 'S'] and podium_data_list:
        podium_cards_cs = []; position_map = {1:"1st Place",2:"2nd Place",3:"3rd Place"}
        default_driver_img_rel_path="images/driver_default.png"; default_team_logo_rel_path="images/team_default.png"
        default_driver_src=app.get_asset_url(default_driver_img_rel_path); default_team_src=app.get_asset_url(default_team_logo_rel_path)
        for driver_info in podium_data_list:
            driver_abbr = driver_info.get('Abbreviation', '').lower(); team_name_simple = driver_info.get('TeamName', 'default').lower().replace(' ', '_').replace('-', '_')
            specific_driver_img_rel_path = f"images/drivers/{driver_abbr}.png"; specific_team_logo_rel_path = f"images/teams/{team_name_simple}.png"
            driver_image_src = default_driver_src
            if driver_abbr and os.path.exists(os.path.join(assets_folder, specific_driver_img_rel_path)): driver_image_src = app.get_asset_url(specific_driver_img_rel_path)
            team_logo_src = default_team_src
            if team_name_simple != 'default' and os.path.exists(os.path.join(assets_folder, specific_team_logo_rel_path)): team_logo_src = app.get_asset_url(specific_team_logo_rel_path)
            card_style = {}; 
            if driver_info.get('TeamColor') and pd.notna(driver_info['TeamColor']): card_style = {'borderLeft': f"5px solid {driver_info['TeamColor']}"}
            podium_cards_cs.append(dbc.Col(dbc.Card([html.Img(src=driver_image_src, className="img-fluid rounded-circle p-2 mx-auto d-block", style={'width': '100px', 'height': '100px', 'objectFit': 'cover'}),
                    dbc.CardBody([html.H5(position_map.get(driver_info['Position'], f"Pos {driver_info['Position']}"), className="card-title"),
                        html.H6(driver_info['DriverName'], className="card-subtitle mb-2"),
                        html.Img(src=team_logo_src, style={'height': '25px', 'marginRight': '5px', 'verticalAlign': 'middle'}),
                        html.Span(driver_info['TeamName'], className="card-text text-muted small")])
                ], className="text-center h-100 shadow-sm", style=card_style), lg=3, md=4, sm=6, className="mb-3"))
        if podium_cards_cs: podium_display_cs = dbc.Row(podium_cards_cs, justify="center")
    if selected_session_type == 'P_OVERVIEW': ai_highlights_md = "AI-generated highlights are not available for Practice Overview sessions."
    elif not raw_results_for_graph.empty:
        summary_parts=[f"Session Data: {actual_session_name} - {selected_event} ({cs_year})"]
        if selected_session_type=='Q':
            if 'Position' in raw_results_for_graph.columns: 
                if not raw_results_for_graph[raw_results_for_graph['Position']==1].empty: pole=raw_results_for_graph[raw_results_for_graph['Position']==1].iloc[0]; summary_parts.append(f"Pole: {pole['DriverName']} ({pole['TeamName']})")
                summary_parts.append("Front Row:"); [summary_parts.append(f"  P{int(r_q['Position'])}: {r_q['DriverName']} ({r_q['TeamName']})") for i_q,r_q in raw_results_for_graph[raw_results_for_graph['Position'].isin([1,2])].sort_values(by='Position').iterrows()]
        elif selected_session_type in ['R','S']:
            if podium_data_list: summary_parts.append("Podium:"); [summary_parts.append(f"  {p['Position']}. {p['DriverName']} ({p['TeamName']})") for p in podium_data_list]
            if 'is_fastest_lap_holder' in display_df.columns: fl_dr_row = display_df[display_df['is_fastest_lap_holder']==True]; 
            if not fl_dr_row.empty: summary_parts.append(f"Fastest Lap: {fl_dr_row.iloc[0]['Driver']}.")
        if selected_session_type in ['Q','R','S']: data_summary_str = "\n".join(summary_parts); ai_highlights_md = get_race_highlights_from_perplexity(data_summary_str)
    else: ai_highlights_md = "Not enough data for AI summary."
    team_color_map_cs = {}
    if not raw_results_for_graph.empty and 'TeamName' in raw_results_for_graph.columns:
        unique_teams_cs = raw_results_for_graph['TeamName'].dropna().unique()
        for team_cs in unique_teams_cs:
            color_cs = None; 
            try: color_cs = ff1_plt.get_team_color(team_cs)
            except: pass
            if color_cs is None and selected_session_type not in ['P_OVERVIEW'] and 'TeamColor' in raw_results_for_graph.columns:
                s = raw_results_for_graph.loc[raw_results_for_graph['TeamName']==team_cs,'TeamColor']
                if not s.empty and pd.notna(s.iloc[0]): cv=str(s.iloc[0]); color_cs=('#'+cv) if not cv.startswith('#') else cv
            team_color_map_cs[team_cs]=color_cs if (color_cs and isinstance(color_cs,str) and color_cs.lower()!="nan") else "#CCCCCC"
    if raw_results_for_graph.empty: fig_cs={'layout':{'title':'No data'}}
    elif selected_session_type == 'P_OVERVIEW':
        if 'DriverName' in raw_results_for_graph.columns:
            pm_df=raw_results_for_graph.melt(id_vars=['DriverName','TeamName'], value_vars=['FP1_Time_seconds','FP2_Time_seconds','FP3_Time_seconds'], var_name='Practice_Session_Stage', value_name='Time_seconds')
            pm_df['Practice_Session_Stage'] = pm_df['Practice_Session_Stage'].str.replace('_Time_seconds',''); pm_df.dropna(subset=['Time_seconds'], inplace=True)
            if not pm_df.empty:fig_cs=px.line(pm_df,x='Practice_Session_Stage',y='Time_seconds',color='DriverName',markers=True,title="Practice Times by Driver",labels={'Practice_Session_Stage':'Session','Time_seconds':'Time(s)'},category_orders={"Practice_Session_Stage":["FP1","FP2","FP3"]}, color_discrete_map=team_color_map_cs); fig_cs.update_layout(legend_title_text='Driver')
            else: fig_cs={'layout':{'title':'No practice times to plot'}}
        else: fig_cs={'layout':{'title':'Practice data error for graph'}}
    elif selected_session_type == 'Q':
        try:
            if 'DriverName' not in raw_results_for_graph.columns: raise ValueError("DriverName missing.")
            id_vars_for_melt = ['DriverName']; value_vars_seconds = []
            if 'TeamName' in raw_results_for_graph.columns: id_vars_for_melt.append('TeamName')
            q_cols = [q for q in ['Q1','Q2','Q3'] if q in raw_results_for_graph.columns]
            if not q_cols: raise ValueError("No Q session columns.")
            q_df = raw_results_for_graph[id_vars_for_melt + q_cols].copy()
            for q_s in q_cols:
                col_sec = q_s + '_seconds'
                if pd.api.types.is_timedelta64_dtype(q_df[q_s]): q_df[col_sec] = q_df[q_s].dt.total_seconds()
                else: q_df[col_sec] = np.nan
                value_vars_seconds.append(col_sec)
            if not value_vars_seconds: raise ValueError("No Q_seconds columns.")
            m_df = q_df.melt(id_vars=id_vars_for_melt, value_vars=value_vars_seconds, var_name='Stage', value_name='Time')
            m_df['Stage'] = m_df['Stage'].str.replace('_seconds',''); m_df.dropna(subset=['Time'], inplace=True)
            if not m_df.empty: fig_cs = px.line(m_df, x='Stage', y='Time', color='DriverName', markers=True, title="Qualifying Times", color_discrete_map=team_color_map_cs if 'TeamName' in id_vars_for_melt else None); fig_cs.update_xaxes(categoryorder='array',categoryarray=['Q1','Q2','Q3']);fig_cs.update_layout(legend_title_text='Driver')
            else: fig_cs = {'layout': {'title': 'No valid quali times'}}
        except Exception as e: print(f"CS Q graph error: {e}"); fig_cs = {'layout': {'title': f'CS Q graph error: {e}'}}
    elif selected_session_type in ['R','S']:
        if 'Points' in raw_results_for_graph.columns and 'DriverName' in raw_results_for_graph.columns:
            graph_df_cs = raw_results_for_graph[['DriverName', 'TeamName', 'Points']].copy()
            graph_df_cs['Points'] = pd.to_numeric(graph_df_cs['Points'], errors='coerce').fillna(0); graph_df_cs = graph_df_cs[graph_df_cs['Points'] > 0]
            if not graph_df_cs.empty: fig_cs = px.bar(graph_df_cs, x='DriverName', y='Points', title=f"Points Scored", color='TeamName', color_discrete_map=team_color_map_cs); fig_cs.update_layout(xaxis_title="Driver", yaxis_title="Points", legend_title_text='Team', xaxis={'categoryorder':'total descending'})
            else: fig_cs = {'layout': {'title': f'No points scored'}}
        else: fig_cs = {'layout': {'title': f'Points data not available'}}
    else: fig_cs={'layout':{'title':'Graph NA'}}
    if selected_session_type in ['R','S'] and isinstance(event_round,(int,np.integer)) and event_round > 0:
        driver_progression_df_cs, constructor_progression_df_cs = get_championship_standings_progression(cs_year, event_round)
        if not driver_progression_df_cs.empty: driver_standings_fig_cs = px.line(driver_progression_df_cs,x='round',y='points',color='driverCode',title=f"Driver Champ ({cs_year})",labels={'round':'Rd','points':'Pts','driverCode':'Driver'})
        else: driver_standings_fig_cs = {'layout':{'title':'Driver Standings: No data'}}
        if not constructor_progression_df_cs.empty:
            constructor_names_cs = constructor_progression_df_cs['constructorName'].unique()
            champ_constructor_color_map_cs = {}
            if constructor_names_cs.size > 0:
                for name_cs in constructor_names_cs:
                    color_cs_champ = team_color_map_cs.get(name_cs) 
                    if not color_cs_champ or color_cs_champ == "#CCCCCC":
                        identifier_to_try_cs = ERGAS_CONSTRUCTOR_NAME_TO_KNOWN_TEAM_ID_OR_COLOR.get(name_cs, name_cs)
                        if isinstance(identifier_to_try_cs, str) and identifier_to_try_cs.startswith("#"): color_cs_champ = identifier_to_try_cs
                        else:
                            try: 
                                specific_constructor_color_cs = ff1_plt.get_team_color(identifier_to_try_cs)
                                if specific_constructor_color_cs: color_cs_champ = specific_constructor_color_cs
                            except: color_cs_champ = None
                    champ_constructor_color_map_cs[name_cs] = color_cs_champ or "#CCCCCC"
            constructor_standings_fig_cs = px.line(constructor_progression_df_cs,x='round',y='points',color='constructorName',title=f"Constructor Champ ({cs_year})",labels={'round':'Rd','points':'Pts','constructorName':'Constructor'},color_discrete_map=champ_constructor_color_map_cs)
        else: constructor_standings_fig_cs = {'layout':{'title':'Constructor Standings: No data'}}
    else: driver_standings_fig_cs, constructor_standings_fig_cs = {'layout':{'title':'Champ NA'}},{'layout':{'title':'Champ NA'}}

    return (page_title_text, podium_display_cs, ai_highlights_md, 
            table_data, table_columns, style_data_conditional_cs, 
            fig_cs, driver_standings_fig_cs, constructor_standings_fig_cs)

# --- REPLACE your existing get_formatted_calendar function with this one ---
def get_formatted_calendar(year):
    print(f"[get_formatted_calendar] Fetching calendar for {year}")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            return dbc.Alert("Race calendar not available yet for this year.", color="warning")

        # Merge with our circuit_data_df to get the image filenames
        schedule_with_images = pd.merge(schedule, circuit_data_df, on='RoundNumber', how='left')

        # Center align the table headers
        header_style = {'textAlign': 'center', 'verticalAlign': 'middle'}
        table_header = [
            html.Thead(html.Tr([
                html.Th("Round", style=header_style),
                html.Th("Grand Prix", style=header_style), 
                html.Th("Country", style=header_style),
                html.Th("Circuit", style=header_style),
                html.Th("Date", style=header_style),
                html.Th("Status", style=header_style)
            ]))
        ]

        table_rows = []
        now_local = pd.Timestamp.now()
        
        cell_style = {'textAlign': 'center', 'verticalAlign': 'middle'}

        for index, row in schedule_with_images.iterrows():
            # Status icon logic
            status_icon = "🏁" if pd.to_datetime(row['EventDate']) < now_local else ""
            
            # Country flag image
            country_flag_path = f"images/flags/{row['Country'].lower().replace(' ', '_')}.png"
            flag_img = html.Img(src=app.get_asset_url(country_flag_path), style={'height': '30px'})
            
            # Mini circuit image
            default_circuit_path = "images/circuit_default.png"
            circuit_image_path = app.get_asset_url(default_circuit_path) 
            if pd.notna(row.get('CircuitImageFile')):
                specific_path = f"images/circuits/{row['CircuitImageFile']}"
                if os.path.exists(os.path.join(assets_folder, specific_path)):
                    circuit_image_path = app.get_asset_url(specific_path)
            circuit_img = html.Img(src=circuit_image_path, style={'height': '35px', 'marginLeft': '10px', 'opacity': '1.0'})

            # Date formatting
            formatted_date = pd.to_datetime(row['EventDate']).strftime('%d %b %Y')

            # Build the table row, applying the cell_style
            table_rows.append(html.Tr([
                html.Td(row['RoundNumber'], style=cell_style),
                html.Td(row['EventName'], style={'verticalAlign': 'middle'}),
                html.Td(flag_img, style=cell_style),
                html.Td([row['Location'], circuit_img], style=cell_style),
                html.Td(formatted_date, style=cell_style),
                html.Td(status_icon, style={**cell_style, 'fontSize': '1.2rem'})
            ]))
        
        table_body = [html.Tbody(table_rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True, responsive=True, className="align-middle")
    except Exception as e:
        print(f"Error fetching calendar: {e}")
        return dbc.Alert(f"Error loading race calendar: {e}", color="danger")

# --- REPLACE your existing render_cs_sub_tab callback with this one ---
@app.callback(
    Output('cs-sub-tab-content-area', 'children'),
    Output('cs-navlink-season-so-far', 'active'),
    Output('cs-navlink-next-race', 'active'),
    Output('cs-navlink-race-calendar', 'active'),
    Output('cs-filters-card', 'style'),
    Input('cs-navlink-season-so-far', 'n_clicks'),
    Input('cs-navlink-next-race', 'n_clicks'),
    Input('cs-navlink-race-calendar', 'n_clicks'),
    State('app-main-tabs', 'active_tab')
)
def render_cs_sub_tab(n_clicks_so_far, n_clicks_next_race, n_clicks_calendar, main_tab_active):
    if main_tab_active != "tab-current-season":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'display': 'block'} 

    ctx = dash.callback_context
    triggered_id = 'cs-navlink-season-so-far' if not ctx.triggered else ctx.triggered[0]['prop_id'].split('.')[0]
        
    active_so_far, active_next_race, active_calendar = False, False, False
    content_to_display, filter_card_style = html.Div(), {'display': 'block'} 

    if triggered_id == 'cs-navlink-season-so-far':
        content_to_display, active_so_far = season_so_far_content_layout, True
    
    elif triggered_id == 'cs-navlink-next-race':
        cs_year = datetime.now().year
        next_race_data = get_next_race_info(cs_year)
        
        if next_race_data:
            # --- Build all sections for the Next Race tab ---
            # Header Section
            header_section = dbc.Container([
                dbc.Row(dbc.Col(html.H4(f"{next_race_data.get('EventName', 'Next Race')}", className="text-center fw-bold"))), html.Hr(),
                dbc.Row([
                    dbc.Col(html.Div(html.Img(src=app.get_asset_url(next_race_data.get('CircuitImageRelativePath')), className="img-fluid rounded track-map-style")), lg=7, md=12, className="mb-3"),
                    dbc.Col(html.Div([
                        dbc.Row([dbc.Col(html.Div([html.Img(src=app.get_asset_url(next_race_data.get('CountryFlagImageRelativePath')), style={'height': '24px', 'marginRight': '10px'}), html.Span(next_race_data.get('CircuitName', 'TBC'), className="h5")]), width=12, className="mb-4")]),
                        dbc.Row([dbc.Col(html.Div([html.P("Race Date", className="text-muted small mb-0"), html.H5(next_race_data.get('RaceDate', 'TBC'))])), dbc.Col(html.Div([html.P("Number of Laps", className="text-muted small mb-0"), html.H5(next_race_data.get('NumberOfLaps', 'TBC'))]))], className="mb-4"),
                        dbc.Row([dbc.Col(html.Div([html.P("Circuit Length", className="text-muted small mb-0"), html.H5(next_race_data.get('CircuitLength', 'TBC'))])), dbc.Col(html.Div([html.P("Race Distance", className="text-muted small mb-0"), html.H5(next_race_data.get('RaceDistance', 'TBC'))]))], className="mb-4"),
                        dbc.Row([dbc.Col(html.Div([html.P("Last Year's Fastest Lap", className="text-muted small mb-0"), html.H6(next_race_data.get('LastYearsFastestLap', 'N/A'), className="fw-bold d-inline-block")]))])
                    ]), lg=5, md=12)
                ], align="center")
            ], fluid=True, className="mb-4")

            # Weather Section
            weather_cards = []
            if next_race_data.get('WeatherData'):
                for day_forecast in next_race_data['WeatherData']:
                    weather_cards.append(dbc.Col(dbc.Card([dbc.CardHeader(day_forecast.get('Date')), dbc.CardBody([html.Img(src=day_forecast.get('Icon'), style={'width': '50px'}), html.H5(day_forecast.get('Temp')), html.P(day_forecast.get('Description'))], className="text-center")])))
            weather_section = dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("Expected Weather Conditions"), dbc.CardBody(dbc.Row(weather_cards) if weather_cards else dbc.Alert("Weather forecast currently unavailable.", color="secondary"))])])], className="mb-4")

            # Schedule Section
            schedule_items = [dbc.ListGroupItem(f"{s.get('Session')}: {s.get('Date')} - {s.get('Time')}") for s in next_race_data.get('SessionSchedule', [])]
            schedule_section = dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("Race Weekend Schedule (Pacific Time)"), dbc.ListGroup(schedule_items, flush=True) if schedule_items else dbc.CardBody(dbc.Alert("Schedule not available.", color="warning"))]))], className="mb-4")

            # Past Winners Section
            winner_cards = []
            if next_race_data.get('PastWinners'):
                for winner in next_race_data['PastWinners']:
                    winner_abbr = winner.get('Abbreviation', '').lower()
                    driver_image_src = app.get_asset_url(f"images/drivers/{winner_abbr}.png") if winner_abbr else app.get_asset_url("images/driver_default.png")
                    winner_cards.append(dbc.Col(dbc.Card([dbc.CardBody([html.H5(f"{winner.get('Year')} Winner"), html.Img(src=driver_image_src, style={'width':'60px', 'height':'60px', 'borderRadius':'50%', 'objectFit':'cover', 'float':'left', 'marginRight':'15px'}), html.H6(winner.get('Winner')), html.P(f"Team: {winner.get('Team')}", className="text-muted"), html.P(f"Best Lap: {winner.get('BestLap')}", className="card-text small")])]), md=4))
            past_winners_section = dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("Recent Winners at this Event"), dbc.CardBody(dbc.Row(winner_cards))]))], className="mb-4")

            # Tyre Strategy & AI Insights Section
            stints_df = next_race_data.get('TyreStrategyData')
            driver_order = next_race_data.get('DriverOrder', [])
            tyre_fig = {'layout': {'title': "Last Year's Tyre Strategy Data Unavailable"}}
            ai_tyre_insights = dbc.Alert("AI Insights require weather and last year's strategy data.", color="info")

            if stints_df is not None and not stints_df.empty:
                # Tyre Graph Plotting
                tyre_fig = go.Figure()
                if driver_order:
                    for driver in driver_order:
                        driver_stints = stints_df[stints_df['Driver'] == driver]
                        for _, stint in driver_stints.iterrows():
                            tyre_fig.add_trace(go.Bar(
                                y=[driver], x=[stint['LapEnd'] - stint['LapStart'] + 1], base=[stint['LapStart']], orientation='h',
                                marker_color=MY_COMPOUND_COLORS.get(stint['Compound'], '#CCCCCC'),
                                text=stint['Compound'], name=stint['Compound'], hoverinfo='text',
                                hovertext=f"Driver: {stint['Driver']}<br>Laps: {stint['LapStart']}-{stint['LapEnd']}<br>Compound: {stint['Compound']}"
                            ))
                    graph_height = max(600, len(driver_order) * 40)
                    tyre_fig.update_layout(title_text="Last Year's Tyre Strategies", xaxis_title="Lap Number", yaxis_title="Driver", barmode='stack', yaxis={'categoryorder':'array', 'categoryarray':driver_order[::-1]}, height=graph_height, showlegend=False, plot_bgcolor='white')
                
                # AI Tyre Strategy Insights Logic
                weather_data = next_race_data.get('WeatherData')
                if weather_data:
                    strategy_summary_list = ["Last Year's Strategy Highlights:"]; strategy_df = stints_df.groupby('Driver')['Compound'].agg(lambda x: ' -> '.join(x)).reset_index(); common_strategies = strategy_df['Compound'].value_counts().head(2)
                    for strategy, count in common_strategies.items(): strategy_summary_list.append(f"  - A {count}-stop ({strategy}) was used by {count} drivers.")
                    weather_summary_list = ["This Year's Weather Forecast:"]; 
                    for day in weather_data:
                        if "Sun" in day.get('Date', ''): weather_summary_list.append(f"  - Race Day ({day['Date']}): {day['Temp']}, {day['Description']}")
                    sc_count = next_race_data.get('SafetyCarsLastYear', 0); sc_summary = f"Safety Cars Last Year: There were {sc_count} safety car deployments."
                    ai_prompt_data = "\n".join(strategy_summary_list) + "\n\n" + "\n".join(weather_summary_list) + "\n" + sc_summary
                    ai_prompt = (f"You are an expert F1 strategist analyzing {next_race_data['EventName']}.\n\n{ai_prompt_data}\n\nBased ONLY on this data, provide 2-3 concise bullet points on potential tyre strategy considerations for this year's race.")
                    ai_tyre_insights = dcc.Markdown(get_race_highlights_from_perplexity(ai_prompt))
            
            tyre_strategy_section = dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("Tyre Strategy Analysis"), dbc.CardBody(dcc.Graph(figure=tyre_fig))]), dbc.Card([dbc.CardHeader("AI Tyre Strategy Insights"), dbc.CardBody(dcc.Loading(ai_tyre_insights))], className="mt-3")])], className="mb-4")
            
            content_to_display = dbc.Container([header_section, weather_section, schedule_section, past_winners_section, tyre_strategy_section], fluid=True)
        else:
            content_to_display = dbc.Alert("Could not load data for the next race. The season may be over.", color="warning")

        active_next_race = True
        filter_card_style = {'display': 'none'}
    
    elif triggered_id == 'cs-navlink-race-calendar':
        cs_year = datetime.now().year
        calendar_table_content = get_formatted_calendar(cs_year)
        content_to_display = dbc.Container([html.H5(f"{cs_year} Race Calendar",className="mt-3 mb-3 text-center"), calendar_table_content], fluid=True)
        active_calendar = True
        filter_card_style = {'display': 'none'}
    
    else: 
        content_to_display, active_so_far = season_so_far_content_layout, True
        
    return content_to_display, active_so_far, active_next_race, active_calendar, filter_card_style

# --- Callbacks for Teams & Drivers Tab ---
# --- REPLACE your existing update_teams_drivers_tab callback with this one ---
@app.callback(
    Output('teams-drivers-content-area', 'children'),
    Input('app-main-tabs', 'active_tab')
)
def update_teams_drivers_tab(active_tab):
    if active_tab != 'tab-teams-drivers':
        return dash.no_update

    cs_year = datetime.now().year
    all_teams_data = get_all_teams_data(cs_year)

    if not all_teams_data:
        return dbc.Alert("Could not load team and driver data for the current season.", color="warning")

    team_cards = []
    for team in all_teams_data:
        # --- Create Driver and Logo Columns ---
        driver_logo_cols = []
        
        # Add Driver #1
        if len(team['Drivers']) > 0:
            driver1 = team['Drivers'][0]
            # Use a default image path if a specific one is missing
            driver1_img_path = f"images/drivers/{driver1.get('Abbreviation', '').lower()}.png"
            driver1_img_src = app.get_asset_url(driver1_img_path) if os.path.exists(os.path.join(assets_folder, driver1_img_path)) else app.get_asset_url("images/driver_default.png")
            
            driver_logo_cols.append(
                dbc.Col(
                    dbc.Card([
                        html.Img(src=driver1_img_src, className="img-fluid rounded-circle p-2 mx-auto d-block", style={'width':'150px', 'height':'150px', 'objectFit':'cover'}),
                        dbc.CardBody(html.H5(driver1['FullName'], className="text-center"))
                    ], style={'backgroundColor': hex_to_rgba(team['TeamColor'], alpha=0.1)}), 
                width=5)
            )

        # Add Team Logo in the middle
        team_logo_simple_name = team['TeamName'].lower().replace(' ', '_').replace('-', '_')
        team_logo_path = f"images/teams/{team_logo_simple_name}.png"
        team_logo_src = app.get_asset_url(team_logo_path) if os.path.exists(os.path.join(assets_folder, team_logo_path)) else app.get_asset_url("images/team_default.png")

        driver_logo_cols.append(
            dbc.Col(
                html.Div(
                    html.Img(src=team_logo_src, style={'width': '150px', 'height': '150px', 'objectFit': 'contain'}),
                    className="d-flex align-items-center justify-content-center h-100"
                ),
                width=2,
                className="d-flex align-items-center justify-content-center"
            )
        )

        # Add Driver #2
        if len(team['Drivers']) > 1:
            driver2 = team['Drivers'][1]
            driver2_img_path = f"images/drivers/{driver2.get('Abbreviation', '').lower()}.png"
            driver2_img_src = app.get_asset_url(driver2_img_path) if os.path.exists(os.path.join(assets_folder, driver2_img_path)) else app.get_asset_url("images/driver_default.png")
            
            driver_logo_cols.append(
                dbc.Col(
                    dbc.Card([
                        html.Img(src=driver2_img_src, className="img-fluid rounded-circle p-2 mx-auto d-block", style={'width':'150px', 'height':'150px', 'objectFit':'cover'}),
                        dbc.CardBody(html.H5(driver2['FullName'], className="text-center"))
                    ], style={'backgroundColor': hex_to_rgba(team['TeamColor'], alpha=0.1)}), 
                width=5)
            )
        
        # --- Create the full card for the team ---
        team_card = dbc.Card([
            dbc.CardHeader(
                html.H4(team['TeamName'], className="mb-0 text-white"), # Only the H4 title remains
                style={'backgroundColor': team['TeamColor']}
            ),
            dbc.CardBody([
                # CORRECTED: Use the 'driver_logo_cols' list we just created
                dbc.Row(driver_logo_cols, className="mb-3", justify="center"), 
                html.Hr(),
                html.H6("AI Insights:", className="mt-3"),
                dcc.Markdown(f"* **Team History:** {team.get('History', 'N/A')}"),
                dcc.Markdown(f"* **2025 Performance:** {team.get('Performance', 'N/A')}")
            ])
        ], className="mb-4 shadow-sm")
        
        team_cards.append(team_card)

    return team_cards

@app.callback(
    Output('predictions-content-area', 'children'),
    Input('app-main-tabs', 'active_tab')
)
def update_predictions_tab(active_tab):
    if active_tab != 'tab-predictions':
        return dash.no_update

    cs_year = datetime.now().year
    sim_results_df = run_reinforcement_simulation(cs_year, 'f1_historical_data.csv', 'f1_prediction_model.joblib')

    if sim_results_df is None or sim_results_df.empty:
        return dbc.Alert("Could not run prediction simulation.", color="danger")
    
    # Build the visualizations
    race_prediction_cards = []
    for _, race_result in sim_results_df.iterrows():
        comparison_df = pd.DataFrame({'P': range(1, 11), 'Predicted Driver': race_result['PredictedTop10'], 'Actual Driver': race_result['ActualTop10']})
        comparison_df['Correct'] = comparison_df.apply(lambda row: "✔️" if row['Predicted Driver'] == row['Actual Driver'] else "❌", axis=1)
        race_card = dbc.Card([dbc.CardHeader(f"Round {race_result['Round']}: {race_result['Race']}"),
            dbc.CardBody(dash_table.DataTable(
                data=comparison_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in comparison_df.columns],
                style_cell={'textAlign': 'center'},
                style_data_conditional=[{'if': {'column_id': 'Correct'}, 'backgroundColor': 'rgba(40, 167, 69, 0.2)', 'fontWeight': 'bold'}]
            ))
        ], className="mb-3")
        race_prediction_cards.append(race_card)

    mae_fig = px.line(sim_results_df, x='Round', y='MAE', title='Model Prediction Error (MAE) Over Season', markers=True)
    mae_fig.update_layout(yaxis_title="Prediction Error (+/- positions)")
    
    last_importances = sim_results_df['FeatureImportances'].iloc[-1]
    imp_df = pd.DataFrame(list(last_importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False).head(10)
    imp_fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title=f"Top 10 Feature Importances (after Round {sim_results_df['Round'].max()})")
    imp_fig.update_layout(yaxis={'categoryorder':'total ascending'})

    return html.Div([
        dbc.Row(dbc.Col(html.H5("2025 Race-by-Race Prediction Simulation"), width=12), className="mb-2"),
        dbc.Row(dbc.Col(race_prediction_cards, width=12), className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=mae_fig), md=6), dbc.Col(dcc.Graph(figure=imp_fig), md=6)], className="mb-4")
    ])

@app.callback(
    Output('predict-race-dropdown', 'options'),
    Input('app-main-tabs', 'active_tab') # Trigger when the Predictions tab is viewed
)
def update_predict_race_dropdown(active_tab):
    if active_tab != 'tab-predictions':
        return []

    print("[update_predict_race_dropdown] Populating dropdown with upcoming races...")
    try:
        cs_year = datetime.now().year
        schedule = ff1.get_event_schedule(cs_year, include_testing=False)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        # Filter for races that are in the future
        upcoming_races = schedule[schedule['EventDate'] > pd.Timestamp.now()].sort_values(by='EventDate')
        
        if upcoming_races.empty:
            return [{'label': 'No upcoming races found for this season.', 'value': ''}]
            
        options = [{'label': f"{row['EventName']} (Round {row['RoundNumber']})", 'value': row['RoundNumber']} 
                   for _, row in upcoming_races.iterrows()]
        
        return options
    except Exception as e:
        print(f"Error populating predict-race-dropdown: {e}")
        return []

def get_features_for_prediction(year, round_number, historical_df):
    """Gathers the latest stats for all current drivers to predict an upcoming race."""
    print(f"[get_features_for_prediction] Gathering features for {year} R{round_number}...")
    try:
        ergast_api = ergast.Ergast()
        
        # 1. Get the last completed race to find the current grid and standings
        schedule = ff1.get_event_schedule(year, include_testing=False)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()]
        if completed_races.empty:
            print("No completed races yet to base prediction on.")
            return pd.DataFrame()
            
        last_race_round = completed_races['RoundNumber'].max()
        session = ff1.get_session(year, last_race_round, 'R'); session.load()
        
        # Get the list of current drivers and teams
        current_grid_df = session.results.loc[:, ['Abbreviation', 'TeamName']].copy()
        current_grid_df.rename(columns={'Abbreviation': 'DriverID', 'TeamName': 'TeamID'}, inplace=True)
        
        # 2. Get latest championship standings
        standings_df = pd.DataFrame()
        try:
            standings_content = ergast_api.get_driver_standings(season=year, round=last_race_round).content
            if standings_content: standings_df = standings_content[0][['driverCode', 'position', 'points']]
        except: pass
        
        # 3. Calculate "form" features from historical + current season data
        all_season_data = pd.concat([historical_df, get_race_data_with_features(year, last_race_round, ergast_api)], ignore_index=True)
        all_season_data.sort_values(by=['Year', 'Round'], inplace=True)
        
        features_list = []
        for _, driver in current_grid_df.iterrows():
            driver_id = driver['DriverID']
            driver_history = all_season_data[all_season_data['DriverID'] == driver_id]
            
            # Calculate form based on the very latest data
            recent_form = driver_history.tail(3)['Points'].mean() if not driver_history.empty else 0
            avg_quali = driver_history['QualifyingPosition'].mean() if not driver_history.empty else 10 # Use 10 as a neutral default
            
            # Get current championship standing
            standing = standings_df[standings_df['driverCode'] == driver_id]
            current_standing = standing['position'].iloc[0] if not standing.empty else 20 # Default to last
            current_points = standing['points'].iloc[0] if not standing.empty else 0
            
            features_list.append({
                'DriverID': driver_id, 'TeamID': driver['TeamID'],
                'QualifyingPosition': avg_quali, # Use average quali as a proxy
                'ChampionshipStanding': current_standing,
                'ChampionshipPoints': current_points,
                'RecentFormPoints': recent_form,
                'TrackID': schedule[schedule['RoundNumber'] == round_number]['Location'].iloc[0],
                'DriverTeamID': f"{driver_id}_{driver['TeamID']}"
            })
            
        return pd.DataFrame(features_list)
    except Exception as e:
        print(f"Error in get_features_for_prediction: {e}"); return pd.DataFrame()

# --- REPLACE your existing predict_next_race callback in app.py ---
@app.callback(
    Output('next-race-prediction-output', 'children'),
    Input('predict-race-button', 'n_clicks'),
    State('predict-race-dropdown', 'value'),
    prevent_initial_call=True
)
def predict_next_race(n_clicks, selected_race_round):
    if not selected_race_round:
        return dbc.Alert("Please select an upcoming race from the dropdown.", color="warning")

    print(f"[predict_next_race] Generating prediction for round {selected_race_round}...")
    cs_year = datetime.now().year
    
    try:
        # 1. Load the trained model and the original training data (to get column structure)
        model = joblib.load('f1_prediction_model.joblib')
        historical_df = pd.read_csv('f1_historical_data.csv')
        
        # 2. Get features for ONLY the current drivers for the upcoming race
        predict_df = get_features_for_prediction(cs_year, selected_race_round, historical_df)
        
        if predict_df.empty:
            return dbc.Alert("Could not gather necessary data to make a prediction. A race may need to be completed in the current season first.", color="danger")

        # 3. Pre-process the prediction data to match the training data format
        X_predict_numerical = predict_df[['QualifyingPosition', 'ChampionshipStanding', 'ChampionshipPoints', 'RecentFormPoints']]
        X_predict_categorical = pd.get_dummies(predict_df[['DriverTeamID', 'TrackID']])
        X_predict = pd.concat([X_predict_numerical, X_predict_categorical], axis=1)

        # Align columns: Ensure the prediction data has the exact same columns as the training data
        training_cols = model.get_booster().feature_names
        X_predict_aligned = X_predict.reindex(columns=training_cols, fill_value=0)
        
        # 4. Make Predictions
        predictions = model.predict(X_predict_aligned)
        predict_df['PredictedPosition'] = predictions
        
        # 5. Display Results
        final_prediction = predict_df.sort_values(by='PredictedPosition').loc[:, ['DriverID', 'TeamID']]
        final_prediction.insert(0, 'P', range(1, len(final_prediction) + 1))
        
        return html.Div([
            html.H5(f"Predicted Top 10 Finishers", className="mt-3"),
            dbc.Table.from_dataframe(final_prediction.head(10), striped=True, bordered=True, hover=True, className="mt-2")
        ])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return dbc.Alert("An error occurred while generating the prediction.", color="danger")

# --- Run the App ---
if __name__ == '__main__':
   app.run(debug=True)