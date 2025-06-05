import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import fastf1 as ff1
from fastf1 import ergast
import fastf1.plotting as ff1_plt 
import pandas as pd
from datetime import datetime
import plotly.express as px
import numpy as np
import os
import google.generativeai as genai # Make sure this is imported

# --- API Key and Global Configs ---
GEMINI_API_KEY = "AIzaSyDCrrKCoOyjjEGdkAwau1hYOSnWV3hxXfM" # <--- REPLACE THIS

if GEMINI_API_KEY == "YOUR_API_KEY_HERE" or GEMINI_API_KEY == "": # Check for empty string too
    print("Warning: GEMINI_API_KEY is not set. AI features will be disabled.")
    GEMINI_API_KEY = None 
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini AI configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini AI: {e}. AI features may not work.")
        GEMINI_API_KEY = None

ERGAS_CONSTRUCTOR_NAME_TO_KNOWN_TEAM_ID_OR_COLOR = {
    "Red Bull": "#0600EF", "Ferrari": "#E80020", "Mercedes": "#27F4D2",
    "McLaren": "#FF8000", "Aston Martin": "#229971", "Sauber": "#00FF00", 
    "Haas F1 Team": "Haas", "RB F1 Team": "#00359F", "Williams": "#64C4FF",
    "Alpine F1 Team":  "#0090FF", "Alpine": "#0090FF" 
}

# --- 1. Configure FastF1 Caching ---
try:
    ff1.Cache.enable_cache('cache')
    print("FastF1 cache enabled.")
except Exception as e:
    print(f"Error enabling FastF1 cache: {e}. Please create a 'cache' folder in your project directory.")
    exit()

# --- HELPER FUNCTION DEFINITIONS ---
def format_timedelta(td_object):
    if pd.isna(td_object): return "N/A"
    if isinstance(td_object, pd.Timedelta):
        total_seconds = td_object.total_seconds(); sign = "-" if total_seconds < 0 else ""; total_seconds = abs(total_seconds)
        hours = int(total_seconds // 3600); minutes = int((total_seconds % 3600) // 60); seconds = total_seconds % 60
        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:06.3f}" if hours > 0 else f"{sign}{minutes:02d}:{seconds:06.3f}"
    return str(td_object)

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

def get_race_highlights_from_gemini(race_summary_data_str):
    print(f"[Gemini] Attempting to generate highlights. Data string length: {len(race_summary_data_str) if race_summary_data_str else 0}")
    if not GEMINI_API_KEY: return "AI highlights unavailable (API key not configured/valid)."
    if not race_summary_data_str: return "Not enough race data for AI highlights."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        prompt = (
            "You are an F1 race analyst. Based on the following F1 session information, "
            "provide 3-4 key highlights of the race in bullet point format. "
            "Focus on interesting outcomes, significant events implied by the data, or notable performances. Be concise and use markdown for bullet points (e.g., * Highlight 1).\n\n"
            "Session Information:\n"
            f"{race_summary_data_str}\n\n"
            "Your 3-4 bullet point highlights:"
        )
        print(f"[Gemini Call] Sending prompt to model 'gemini-1.5-flash-latest' (first 100 chars): {prompt[:100]}...")
        response = model.generate_content(prompt)
        print("[Gemini Call] Received response.")
        if response.parts:
            generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            if generated_text: print(f"[Gemini Call] Generated text (first 100 chars): {generated_text[:100]}..."); return generated_text
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason; block_reason_message = response.prompt_feedback.block_reason_message or "No specific message."
            print(f"[Gemini Call] Content blocked. Reason: {block_reason}, Message: {block_reason_message}")
            return f"AI highlights could not be generated. Reason: Content blocked by API ({block_reason})."
        return "AI highlights could not be generated (empty or unexpected response from AI)."
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e): return "AI highlights unavailable: API key/permission issue."
        return f"Error generating AI highlights: {str(e)[:200]}"

# --- MAIN DATA FETCHING FUNCTION (get_session_results) --- (MODIFIED for P_OVERVIEW table)
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

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
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
    html.H5("Next Race Details", className="mt-3 mb-3 text-center"),
    dbc.Alert("Information about the next upcoming race will be displayed here.", color="info")
    # We will add components like event details, circuit map, session times later
], fluid=True, id="next-race-container") # Add ID for clarity

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

# --- Placeholder content for other tabs ---
tab_tracks_content = dbc.Card(dbc.CardBody([html.P("Content for Tracks and Race Strategies will be built here.")]), className="mt-3")
tab_teams_drivers_content = dbc.Card(dbc.CardBody([html.P("Content for Teams and Drivers analysis will be built here.")]), className="mt-3")
tab_predictions_content = dbc.Card(dbc.CardBody([html.P("Content for Predictions models and results will be built here.")]), className="mt-3")
tab_fantasy_rules_content = dbc.Card(dbc.CardBody([html.P("Content for Fantasy League Rules and Restrictions will be built here.")]), className="mt-3")
tab_fantasy_creator_content = dbc.Card(dbc.CardBody([html.P("Content for Fantasy Team Creator tool will be built here.")]), className="mt-3")

# --- Main App Layout ---
app.layout = dbc.Container([
    dcc.Store(id='cs-active-selection-store'),
    dbc.Row(dbc.Col(html.H1("F1 Insights & Fantasy Predictor", className="page-main-title"), width=12), className="mb-3 mt-3 text-center"),
    dbc.Tabs([
        dbc.Tab(tab_historical_content, label="Past Seasons", tab_id="tab-historical"),
        dbc.Tab(tab_2025_content, label=f"Current Season ({current_year})", tab_id="tab-current-season"),
        dbc.Tab(tab_tracks_content, label="Tracks & Strategies", tab_id="tab-tracks"),
        dbc.Tab(tab_teams_drivers_content, label="Teams & Drivers", tab_id="tab-teams-drivers"),
        dbc.Tab(tab_predictions_content, label="Predictions", tab_id="tab-predictions"),
        dbc.Tab(tab_fantasy_rules_content, label="Fantasy Rules", tab_id="tab-fantasy-rules"),
        dbc.Tab(tab_fantasy_creator_content, label="Fantasy Team Creator", tab_id="tab-fantasy-creator"),
    ], id="app-main-tabs", active_tab="tab-historical"),
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
        race_data_summary_hist_str = "\n".join(summary_parts_hist); ai_highlights_hist_md = get_race_highlights_from_gemini(race_data_summary_hist_str)
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
    triggered_id = dash.callback_context.triggered_id
    is_initial_load_on_cs_tab = triggered_id is None and active_main_tab == "tab-current-season"
    is_tab_switch_to_cs = triggered_id == 'app-main-tabs' and active_main_tab == "tab-current-season"
    
    if not (is_initial_load_on_cs_tab or is_tab_switch_to_cs):
        return dash.no_update, dash.no_update

    cs_year = datetime.now().year 
    print(f"[Callback cs_event_dropdown] Populating for Current Season ({cs_year})")
    
    try:
        full_schedule = ff1.get_event_schedule(cs_year, include_testing=False)
        if full_schedule.empty: 
            print(f"[Callback cs_event_dropdown] No events found in schedule for {cs_year}.")
            return [], None
        
        # Ensure EventDate is datetime and localized to UTC for proper comparison
        # FastF1 event dates are usually naive, representing local time of the event.
        # For comparing with 'now', it's safer to assume they are event local times.
        # pd.Timestamp.now() is naive by default (local system time).
        # For simplicity, if ff1 returns naive dates, we compare with naive now.
        full_schedule['EventDate'] = pd.to_datetime(full_schedule['EventDate'])
        now_local = pd.Timestamp.now() # Naive local time

        # Filter for completed events (whose date is in the past)
        completed_events_schedule = full_schedule[full_schedule['EventDate'] < now_local].copy()
        
        options_df = pd.DataFrame()
        if not completed_events_schedule.empty:
            print(f"[Callback cs_event_dropdown] Found {len(completed_events_schedule)} completed events for {cs_year}.")
            options_df = completed_events_schedule.sort_values(by='RoundNumber').drop_duplicates(subset=['EventName'], keep='first')
        else: # No events completed yet, show all scheduled events for selection, default to first
            print(f"[Callback cs_event_dropdown] No completed events yet for {cs_year}. Showing full schedule.")
            options_df = full_schedule.sort_values(by='RoundNumber').drop_duplicates(subset=['EventName'], keep='first')

        if options_df.empty:
            print(f"[Callback cs_event_dropdown] No events available to populate dropdown for {cs_year}.")
            return [], None
            
        cs_event_options = [{'label': f"{row['EventName']} (R{row['RoundNumber']})", 'value': row['EventName']} 
                             for idx, row in options_df.iterrows()]
        
        default_event = None
        if not completed_events_schedule.empty:
            # Default to the latest completed event (highest round number among completed)
            default_event = completed_events_schedule.sort_values(by='RoundNumber', ascending=False).iloc[0]['EventName']
        elif cs_event_options: 
            # If no events completed, default to the first event of the season (Round 1)
            default_event = cs_event_options[0]['value'] 
            
        print(f"[Callback cs_event_dropdown] Options generated: {len(cs_event_options)}. Default event: {default_event}")
        return cs_event_options, default_event
    except Exception as e:
        print(f"Error in cs_event_dropdown: {e}")
        return [], None

# --- update_cs_season_so_far_content Callback (MODIFIED with debug print) ---
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
        if selected_session_type in ['Q','R','S']: data_summary_str = "\n".join(summary_parts); ai_highlights_md = get_race_highlights_from_gemini(data_summary_str)
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

def get_formatted_calendar(year):
    print(f"[get_formatted_calendar] Fetching calendar for {year}")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            return dbc.Alert("Race calendar not available yet for this year.", color="warning")
        
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.strftime('%Y-%m-%d (%A)')
        schedule_for_table = schedule[['RoundNumber', 'EventName', 'Country', 'Location', 'EventDate']]
        schedule_for_table.rename(columns={
            'RoundNumber': 'Round', 'EventName': 'Event', 'Location': 'Circuit'
        }, inplace=True)

        table = dbc.Table.from_dataframe(schedule_for_table, striped=True, bordered=True, hover=True, responsive=True)
        return table
    except Exception as e:
        print(f"Error fetching calendar: {e}")
        return dbc.Alert(f"Error loading race calendar: {e}", color="danger")

@app.callback(
    Output('cs-sub-tab-content-area', 'children'),
    Output('cs-navlink-season-so-far', 'active'),
    Output('cs-navlink-next-race', 'active'),
    Output('cs-navlink-race-calendar', 'active'),
    Output('cs-filters-card', 'style'),  # NEW OUTPUT to control filter visibility
    Input('cs-navlink-season-so-far', 'n_clicks'),
    Input('cs-navlink-next-race', 'n_clicks'),
    Input('cs-navlink-race-calendar', 'n_clicks'),
    State('app-main-tabs', 'active_tab')
)
def render_cs_sub_tab(n_clicks_so_far, n_clicks_next_race, n_clicks_calendar, main_tab_active):
    if main_tab_active != "tab-current-season":
        # Default style for filters if this tab is not active (should remain visible if no other logic hides it)
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'display': 'block'} 

    ctx = dash.callback_context
    # Default to "Season So Far" if no clicks yet or on initial load of this tab
    triggered_id = 'cs-navlink-season-so-far' 
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
    active_so_far, active_next_race, active_calendar = False, False, False
    content_to_display = html.Div() 
    filter_card_style = {'display': 'block'} # Default to show filters

    if triggered_id == 'cs-navlink-season-so-far':
        content_to_display = season_so_far_content_layout # Your layout for this sub-tab
        active_so_far = True
        filter_card_style = {'display': 'block'} # Show filters
    elif triggered_id == 'cs-navlink-next-race':
        content_to_display = next_race_content_layout # Your placeholder layout
        active_next_race = True
        filter_card_style = {'display': 'none'} # Hide filters
    elif triggered_id == 'cs-navlink-race-calendar':
        cs_year = datetime.now().year
        calendar_table_content = get_formatted_calendar(cs_year)
        content_to_display = dbc.Container([
            html.H5(f"{cs_year} Race Calendar", className="mt-3 mb-3 text-center"),
            calendar_table_content
        ], fluid=True, id="race-calendar-container-content")
        active_calendar = True
        filter_card_style = {'display': 'none'} # Hide filters
    else: # Default case
        content_to_display = season_so_far_content_layout
        active_so_far = True
        filter_card_style = {'display': 'block'} # Show filters for default
        
    return content_to_display, active_so_far, active_next_race, active_calendar, filter_card_style

# --- Run the App ---
if __name__ == '__main__':
   app.run(debug=True)