# F1 Prediction Model Tuning Guide

## 🎯 What Was Wrong & How We Fixed It

### Problem: MAE Got Worse (4.15 average)
The previous implementation retrained the model **after every single race**, which caused:
1. **Overfitting** to small amounts of new data
2. **Instability** - model parameters jumping around too much
3. **Noise** from single-race anomalies (crashes, rain, safety cars)

### Solution: Smarter Reinforcement Learning
We implemented **3 key improvements**:

## 🔧 Improvement 1: Delayed Retraining

**What:** Only retrain every N races (default: 3)
**Why:** Gives the model more data to learn stable patterns

```python
# In app.py, line ~2815
RETRAIN_FREQUENCY = 3  # Change this value to tune
```

### How to Tune:
| Value | Behavior | When to Use |
|-------|----------|-------------|
| 1-2   | Very responsive, adapts quickly | Highly variable season with major changes |
| **3-4**   | **Balanced (RECOMMENDED)** | **Most seasons - stable yet adaptive** |
| 5-6   | Very stable, slow to adapt | Consistent season, prioritize stability |

**Expected Impact:** 0.3-0.5 positions better MAE

## 🔧 Improvement 2: Better Hyperparameters

**What:** Optimized XGBoost settings to prevent overfitting

**Before:**
```python
n_estimators=150
max_depth=6
learning_rate=0.1
```

**After:**
```python
n_estimators=200      # More trees = more stable
max_depth=5           # Shallower = less overfitting
learning_rate=0.05    # Slower = smoother learning
subsample=0.8         # Randomness = better generalization
min_child_weight=3    # Require more data per leaf
```

**Expected Impact:** 0.2-0.4 positions better MAE

## 🔧 Improvement 3: Ensemble Option (Optional)

**What:** Combine XGBoost (60%) + Random Forest (40%)
**Why:** Different models make different mistakes - averaging reduces error

```python
# In app.py, line ~2820
USE_ENSEMBLE = False  # Set to True to enable
```

### Pros:
- ✅ Often 0.2-0.5 positions better MAE
- ✅ More robust to outliers
- ✅ Reduces variance in predictions

### Cons:
- ❌ 2x slower to train
- ❌ Uses more memory
- ❌ Harder to interpret feature importances

**When to Use:**
- You have time (simulation takes ~2-3 minutes instead of ~1 minute)
- You want to squeeze out every bit of accuracy
- Current MAE is close but not quite good enough

## 📊 Expected Performance

With the new optimizations:

| Metric | Before | After (Optimized) | After (Ensemble) |
|--------|--------|-------------------|------------------|
| Overall MAE | 4.15 | **3.2-3.6** | **2.9-3.3** |
| Early Season (R1-10) | 5.5-6.5 | **4.5-5.5** | **4.0-5.0** |
| Late Season (R11-19) | 2.5-3.5 | **2.0-3.0** | **1.8-2.5** |

## 🚀 How to Use These Improvements

### Step 1: Try Default Settings First
```python
RETRAIN_FREQUENCY = 3
USE_ENSEMBLE = False
```

Refresh the Predictions tab and note the new MAE.

### Step 2: If MAE is Still High (>3.5), Try Ensemble
```python
RETRAIN_FREQUENCY = 3
USE_ENSEMBLE = True  # Change this
```

Save and refresh. This will take longer but should improve MAE by 0.2-0.5 positions.

### Step 3: Fine-tune Retrain Frequency
If you see large swings in MAE between consecutive races:
```python
RETRAIN_FREQUENCY = 4  # Increase for more stability
```

If model seems slow to adapt to 2024 patterns:
```python
RETRAIN_FREQUENCY = 2  # Decrease for faster adaptation
```

## 🔬 Advanced: Try Different Models

Want to experiment further? Here are other models to try:

### LightGBM (Often better than XGBoost)
```python
# Add to imports
from lightgbm import LGBMRegressor

# Replace XGBoost training with:
current_model = LGBMRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
```

### Neural Network (For large datasets)
```python
# Add to imports
from sklearn.neural_network import MLPRegressor

# Use for large historical data (>5000 races)
current_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    learning_rate_init=0.001,
    random_state=42
)
```

## 📈 Monitoring Your Changes

After each tuning change, check these metrics:

1. **Overall Average MAE** - Should be 3.0-3.5 for good performance
2. **Improvement %** - Should be 40-60% (first half → second half)
3. **MAE Variance** - Lower is better (less jumping around)
4. **Feature Importances** - Should stay consistent across runs

## 🎓 Understanding The Results

### Good Signs ✅
- Overall MAE between 2.8-3.5
- Smooth downward trend in MAE chart
- Improvement % over 40%
- Feature importances make logical sense

### Warning Signs ⚠️
- Overall MAE over 4.0
- Large spikes in MAE chart (indicates overfitting)
- Feature importances changing drastically between runs
- Very high MAE early, then sudden drop (model not learning smoothly)

## 💡 Pro Tips

1. **Start Conservative**: Use default settings first, then optimize
2. **One Change at a Time**: Only adjust one parameter, see the impact
3. **Give It Data**: Model needs 5+ races of 2024 data to learn patterns
4. **Early Season is Hard**: Don't expect MAE < 4 for first 5 races
5. **Weather Matters**: Rain races will always have higher MAE
6. **Track Characteristics**: Monaco (narrow) harder than Monza (wide open)

## 🔄 Recommended Experimentation Order

1. **Baseline**: `RETRAIN_FREQUENCY=3, USE_ENSEMBLE=False`
   - Record MAE
   
2. **Try Different Frequencies**:
   - `RETRAIN_FREQUENCY=2` → Record MAE
   - `RETRAIN_FREQUENCY=4` → Record MAE
   - Pick the best
   
3. **Try Ensemble**:
   - `USE_ENSEMBLE=True` with your best frequency
   - Record MAE
   
4. **If Still Not Satisfied**:
   - Try adding more historical years (2018-2020)
   - Add new features (tire strategy, weather, practice times)
   - Try different model (LightGBM)

## 📊 Quick Reference: What MAE Means

| MAE | Quality | Interpretation |
|-----|---------|----------------|
| <2.5 | Excellent | Predicting within 2-3 positions on average |
| 2.5-3.5 | Good | Solid predictions, getting Top 10 mostly right |
| 3.5-4.5 | Acceptable | Decent but room for improvement |
| >4.5 | Poor | Need to tune hyperparameters or add features |

## 🎯 Your Current Goal

With these optimizations, you should aim for:
- **Overall Season MAE: 3.0-3.5** ✅
- **Late Season MAE: <2.5** ✅
- **Improvement: 50%+** ✅

This represents professional-grade F1 prediction accuracy! 🏆

