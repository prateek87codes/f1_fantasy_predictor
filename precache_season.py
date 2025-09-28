# precache_season.py
import fastf1 as ff1
import os
from tqdm import tqdm

def run_precache(year):
    """Downloads and caches essential data for all GPs in a given year."""
    
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    ff1.Cache.enable_cache(cache_dir)
    print(f"FastF1 cache enabled at: {cache_dir}")
    
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        # Filter for actual Grand Prix events
        races = schedule[schedule['EventName'].str.contains('Grand Prix')]
        
        print(f"\n--- Starting to precache data for {year} season ({len(races)} events) ---")
        
        for _, event in tqdm(races.iterrows(), total=len(races), desc="Caching Events"):
            try:
                print(f"\nCaching: {event['EventName']}")
                # Load only the most essential data used by your app
                session = ff1.get_session(year, event['EventName'], 'R')
                session.load(laps=True, telemetry=False, weather=True, messages=True)
                print(f"Successfully cached Race session for {event['EventName']}.")
            except Exception as e:
                print(f"Could not cache {event['EventName']}. Error: {e}")
                
        print("\n--- Pre-caching complete! ---")
        
    except Exception as e:
        print(f"An error occurred during the precache process: {e}")

if __name__ == "__main__":
    # You can change the year to precache different seasons
    run_precache(2024)