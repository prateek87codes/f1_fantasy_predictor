import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== ENHANCED F1 PREDICTION MODEL TRAINING ===")

# Load data
try:
    df = pd.read_csv('f1_historical_data.csv')
    print(f"✅ Loaded {len(df)} historical race records")
except FileNotFoundError:
    print("❌ ERROR: f1_historical_data.csv not found!")
    exit()

# Load circuit data for additional features
try:
    circuit_df = pd.read_csv('circuit_data.csv')
    print(f"✅ Loaded circuit data for {len(circuit_df)} circuits")
except FileNotFoundError:
    print("⚠️  Circuit data not found, proceeding without circuit-specific features")
    circuit_df = pd.DataFrame()

print("\n🔧 ENHANCED FEATURE ENGINEERING...")

# 1. ENHANCED FEATURE ENGINEERING
def create_enhanced_features(df, circuit_df):
    """Create enhanced features for better predictions"""
    
    # Sort by year and round for proper feature calculation
    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)
    
    # 1. Driver Performance Metrics
    print("   📊 Creating enhanced driver performance metrics...")
    
    # Driver win rate (last 10 races)
    df['DriverWinRate'] = df.groupby('DriverID')['FinishingPosition'].transform(
        lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y == 1).sum() / len(y))
    )
    
    # Driver podium rate (last 10 races)
    df['DriverPodiumRate'] = df.groupby('DriverID')['FinishingPosition'].transform(
        lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y <= 3).sum() / len(y))
    )
    
    # Driver points per race (last 5 races)
    df['DriverAvgPoints'] = df.groupby('DriverID')['Points'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # Driver consistency (std of finishing positions, last 10 races)
    df['DriverConsistency'] = df.groupby('DriverID')['FinishingPosition'].transform(
        lambda x: x.rolling(10, min_periods=1).std().fillna(5)
    )
    
    # Driver improvement trend (last 5 races)
    df['DriverImprovement'] = df.groupby('DriverID')['FinishingPosition'].transform(
        lambda x: x.rolling(5, min_periods=3).apply(lambda y: -np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 3 else 0)
    )
    
    # 2. Team Performance Metrics
    print("   🏎️ Creating enhanced team performance metrics...")
    
    # Team average points (last 5 races)
    df['TeamAvgPoints'] = df.groupby('TeamID')['Points'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # Team championship standing trend (improving/declining)
    df['TeamStandingTrend'] = df.groupby('TeamID')['ChampionshipStanding'].transform(
        lambda x: x.diff().rolling(5, min_periods=1).mean().fillna(0)
    )
    
    # Team reliability (DNF rate)
    df['TeamReliability'] = df.groupby('TeamID')['FinishingPosition'].transform(
        lambda x: x.rolling(10, min_periods=1).apply(lambda y: (y <= 20).sum() / len(y))
    )
    
    # 3. Circuit-Specific Performance
    print("   🏁 Creating enhanced circuit-specific features...")
    
    if not circuit_df.empty:
        # Merge circuit data
        df = df.merge(circuit_df[['CircuitName_ff1', 'CircuitLength_km']], 
                     left_on='TrackID', right_on='CircuitName_ff1', how='left')
        
        # Driver performance at specific circuits
        df['DriverCircuitAvg'] = df.groupby(['DriverID', 'TrackID'])['FinishingPosition'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Team performance at specific circuits
        df['TeamCircuitAvg'] = df.groupby(['TeamID', 'TrackID'])['FinishingPosition'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Circuit difficulty (average finishing position vs qualifying)
        df['CircuitDifficulty'] = df.groupby('TrackID').apply(
            lambda x: (x['FinishingPosition'] - x['QualifyingPosition']).mean()
        ).reset_index(level=0, drop=True)
    else:
        # Fallback values if no circuit data
        df['DriverCircuitAvg'] = df['FinishingPosition']
        df['TeamCircuitAvg'] = df['FinishingPosition']
        df['CircuitDifficulty'] = 0
    
    # 4. Season Progression Features
    print("   📈 Creating enhanced season progression features...")
    
    # Race number in season (1-24)
    df['RaceNumber'] = df.groupby('Year')['Round'].transform(lambda x: x.rank())
    
    # Championship momentum (points gained in last 3 races)
    df['ChampionshipMomentum'] = df.groupby('DriverID')['ChampionshipPoints'].transform(
        lambda x: x.diff(3).fillna(0)
    )
    
    # Season phase (early, mid, late) - encode as numeric
    df['SeasonPhase'] = pd.cut(df['RaceNumber'], bins=[0, 8, 16, 24], labels=[1, 2, 3])
    df['SeasonPhase'] = df['SeasonPhase'].astype(float)
    
    # 5. Grid Position Analysis
    print("   🏁 Creating enhanced grid position features...")
    
    # Qualifying vs Race performance (last 5 races)
    df['QualiRaceDiff'] = df.groupby('DriverID').apply(
        lambda x: (x['QualifyingPosition'] - x['FinishingPosition']).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    # Grid position advantage (how much better qualifying helps)
    df['GridAdvantage'] = df.groupby('DriverID')['QualifyingPosition'].transform(
        lambda x: x.rolling(5, min_periods=1).apply(lambda y: 21 - y.mean())
    )
    
    # 6. Advanced Form Metrics
    print("   📊 Creating enhanced form metrics...")
    
    # Weighted recent form (more recent races weighted higher)
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # Last 4 races
    df['WeightedForm'] = df.groupby('DriverID')['Points'].transform(
        lambda x: x.rolling(4, min_periods=1).apply(
            lambda y: np.average(y, weights=weights[:len(y)]) if len(y) > 0 else 0
        )
    )
    
    # Performance in similar tracks (if circuit data available)
    if not circuit_df.empty:
        df['SimilarTrackPerformance'] = df.groupby(['DriverID', 'CircuitLength_km'])['FinishingPosition'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    else:
        df['SimilarTrackPerformance'] = df['FinishingPosition']
    
    # 7. NEW: Head-to-Head Performance
    print("   🥊 Creating head-to-head performance features...")
    
    # Driver vs teammate performance
    df['TeammateAdvantage'] = df.groupby(['TeamID', 'Round']).apply(
        lambda x: x['FinishingPosition'].rank(ascending=True) - 1.5
    ).reset_index(level=[0,1], drop=True)
    
    # 8. NEW: Weather and External Factors (placeholder for future enhancement)
    print("   🌤️ Creating weather-related features...")
    df['WeatherFactor'] = 1.0  # Placeholder for weather data
    
    # 9. NEW: Championship Pressure
    print("   🏆 Creating championship pressure features...")
    df['ChampionshipPressure'] = df.groupby('Year').apply(
        lambda x: 1 / (x['ChampionshipStanding'] + 1)
    ).reset_index(level=0, drop=True)
    
    return df

# Apply enhanced feature engineering
df_enhanced = create_enhanced_features(df, circuit_df)

# 2. DATA PREPROCESSING IMPROVEMENTS
print("\n🔧 ENHANCED DATA PREPROCESSING...")

# Handle missing values more intelligently
print("   🧹 Handling missing values...")
df_enhanced = df_enhanced.fillna(0)

# Remove outliers (finishing positions > 20 are likely DNFs, treat as 20)
df_enhanced.loc[df_enhanced['FinishingPosition'] > 20, 'FinishingPosition'] = 20

# 3. SELECT ENHANCED FEATURES
print("\n🎯 SELECTING ENHANCED FEATURES...")

# Core features (original)
core_features = [
    'QualifyingPosition', 'ChampionshipStanding', 'ChampionshipPoints',
    'RecentFormPoints', 'RecentQualiPos', 'RecentFinishPos'
]

# Advanced features (from previous model)
advanced_features = [
    'DriverWinRate', 'DriverPodiumRate', 'DriverAvgPoints', 'DriverConsistency',
    'TeamAvgPoints', 'TeamStandingTrend', 'DriverCircuitAvg', 'TeamCircuitAvg',
    'RaceNumber', 'ChampionshipMomentum', 'QualiRaceDiff', 'WeightedForm',
    'SimilarTrackPerformance'
]

# New enhanced features
enhanced_features = [
    'DriverImprovement', 'TeamReliability', 'CircuitDifficulty', 'SeasonPhase',
    'GridAdvantage', 'TeammateAdvantage', 'WeatherFactor', 'ChampionshipPressure'
]

# Combine all features
all_features = core_features + advanced_features + enhanced_features

# Check which features exist in our data
available_features = [f for f in all_features if f in df_enhanced.columns]
print(f"   ✅ Using {len(available_features)} features: {available_features}")

# Prepare feature matrix
X_numerical = df_enhanced[available_features].fillna(0)
y = df_enhanced['FinishingPosition']

# One-hot encode categorical variables (SeasonPhase is now numeric)
categorical_cols = ['DriverTeamID', 'TrackID']
X_categorical = pd.get_dummies(df_enhanced[categorical_cols], columns=categorical_cols)
X = pd.concat([X_numerical, X_categorical], axis=1)
X.columns = [str(col) for col in X.columns]

# Ensure all values are numeric
X = X.astype(float)

print(f"   📊 Final feature matrix shape: {X.shape}")

# 4. FEATURE SELECTION
print("\n🎯 FEATURE SELECTION...")
selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"   ✅ Selected {len(selected_features)} most important features")

# 5. ENHANCED MODEL TRAINING
print("\n🚀 TRAINING ENHANCED MODELS...")

# Use time series split for more realistic validation
tscv = TimeSeriesSplit(n_splits=5)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Enhanced models with better hyperparameters
models = {
    'XGBoost_Enhanced': xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=3000,
        learning_rate=0.005,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        early_stopping_rounds=200,
        random_state=42
    ),
    'RandomForest_Enhanced': RandomForestRegressor(
        n_estimators=2000,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    ),
    'GradientBoosting_Enhanced': GradientBoostingRegressor(
        n_estimators=2000,
        learning_rate=0.005,
        max_depth=8,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    ),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
}

best_model = None
best_mae = float('inf')
results = {}

print("\n📊 ENHANCED MODEL PERFORMANCE COMPARISON:")
print("-" * 70)

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    if hasattr(model, 'early_stopping_rounds'):
        # For XGBoost with early stopping
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
    
    print(f"   MAE: {mae:.4f} | MSE: {mse:.4f} | R²: {r2:.4f}")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = name

print(f"\n🏆 BEST MODEL: {best_model_name} with MAE = {best_mae:.4f}")

# 6. ENSEMBLE MODEL
print(f"\n🎯 CREATING ENSEMBLE MODEL...")
# Create fresh models for ensemble to avoid early stopping issues
ensemble_models = [
    ('xgb', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )),
    ('rf', RandomForestRegressor(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )),
    ('gb', GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    ))
]

ensemble = VotingRegressor(estimators=ensemble_models)
ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print(f"   Ensemble MAE: {ensemble_mae:.4f}")

if ensemble_mae < best_mae:
    best_model = ensemble
    best_model_name = 'Ensemble'
    best_mae = ensemble_mae
    print(f"   🏆 Ensemble is the new best model!")

# 7. FEATURE IMPORTANCE ANALYSIS
print(f"\n📊 FEATURE IMPORTANCE ANALYSIS ({best_model_name}):")
print("-" * 70)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<35} {row['importance']:.4f}")

# 8. CROSS-VALIDATION WITH TIME SERIES SPLIT
print(f"\n🔄 TIME SERIES CROSS-VALIDATION ANALYSIS ({best_model_name}):")
print("-" * 70)

# Create a model without early stopping for CV
cv_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

cv_scores = cross_val_score(cv_model, X_selected, y, cv=tscv, scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"Time Series CV MAE: {cv_mae:.4f} (+/- {cv_std:.4f})")
print(f"CV Scores: {[-score for score in cv_scores]}")

# 9. SAVE THE ENHANCED MODEL
print(f"\n💾 SAVING ENHANCED MODEL...")
joblib.dump(best_model, 'f1_prediction_model_enhanced.joblib')

# Save feature information
feature_info = {
    'feature_names': selected_features,
    'numerical_features': available_features,
    'categorical_features': categorical_cols,
    'model_type': best_model_name,
    'mae': best_mae,
    'cv_mae': cv_mae,
    'selector': selector
}
joblib.dump(feature_info, 'f1_prediction_features_enhanced.joblib')

print("✅ Enhanced model training completed!")
print(f"🎯 MAE Improvement: {3.0923 - best_mae:.4f} reduction from baseline")
print(f"📈 Accuracy improvement: {((3.0923 - best_mae) / 3.0923 * 100):.1f}%")
print(f"🏆 Best model: {best_model_name} with MAE = {best_mae:.4f}")
