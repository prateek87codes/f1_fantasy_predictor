import fastf1
import pandas as pd

# This option ensures pandas prints all rows of the DataFrame
pd.set_option('display.max_rows', None)

print("--- Fetching 2025 F1 Schedule to Get Official Country Names ---")

try:
    # Get the schedule for the year 2025
    schedule_2025 = fastf1.get_event_schedule(2025, include_testing=False)

    if not schedule_2025.empty:
        # Get unique country names to avoid duplicates, while keeping event info
        unique_events = schedule_2025.drop_duplicates(subset=['EventName']).sort_values(by='RoundNumber')

        # Create a new DataFrame for clean output
        country_info = pd.DataFrame({
            'Round': unique_events['RoundNumber'],
            'Event Name': unique_events['EventName'],
            'Country Name (from API)': unique_events['Country'],
            'Required Filename': unique_events['Country'].str.lower().str.replace(' ', '_') + '.png'
        })
        
        # Print the formatted table to the terminal
        print(country_info.to_string(index=False))
        print("\nNOTE: Use the 'Required Filename' for your image files inside 'assets/images/flags/'")
        
    else:
        print("Could not fetch the 2025 schedule.")

except Exception as e:
    print(f"An error occurred: {e}")