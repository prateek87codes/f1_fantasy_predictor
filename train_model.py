import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

print("--- Starting Model Training ---")

# 1. Load the Historical Data
try:
    df = pd.read_csv('f1_historical_data.csv')
    print(f"Successfully loaded f1_historical_data.csv with {len(df)} rows.")
except FileNotFoundError:
    print("ERROR: f1_historical_data.csv not found. Please run the data_gathering.py script first.")
    exit()

# 2. Feature Engineering & Preprocessing
print("Preparing data for training...")

# Define features (X) and target (y)
# We now use our new, more powerful features!
features = [
    'QualifyingPosition', 
    'ChampionshipStanding', 
    'ChampionshipPoints', 
    'RecentFormPoints',
    'RecentQualiPos',
    'RecentFinishPos'
]
target = 'FinishingPosition'

X_numerical = df[features]
y = df[target]

# One-Hot Encode our combined DriverTeamID and the TrackID
X_categorical = pd.get_dummies(df[['DriverTeamID', 'TrackID']], columns=['DriverTeamID', 'TrackID'])
X = pd.concat([X_numerical, X_categorical], axis=1)
X.columns = [str(col) for col in X.columns]

# 3. Split, Train, Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training XGBoost model with new 'Form' features...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, early_stopping_rounds=50, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE) with new features: {mae:.4f}")

# 4. Save the new, smarter model
print("\nSaving new trained model to 'f1_prediction_model.joblib'...")
joblib.dump(model, 'f1_prediction_model.joblib')
print("--- Model Saved Successfully! ---")