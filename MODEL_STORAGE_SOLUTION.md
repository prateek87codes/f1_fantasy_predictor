# Model Storage Solution for Vercel Deployment

## Problem
Vercel has a 300MB deployment limit. The project exceeds this due to:
- Large ML model file (18MB)
- Heavy dependencies (xgboost, scikit-learn, etc.)

## Solution Options

### Option 1: Store Model on GitHub Releases (Recommended)
1. Upload `f1_prediction_model_enhanced.joblib` to GitHub Releases
2. Download it at runtime in the app
3. Cache it in `/tmp` (Vercel's writable directory)

### Option 2: Use External Storage (S3, Cloud Storage)
- Upload model to S3/Google Cloud Storage
- Download at runtime
- More reliable but requires setup

### Option 3: Use Git LFS
- Store model in Git LFS
- Vercel should handle it automatically
- Requires Git LFS setup

## Quick Fix: Exclude Model from Deployment

For now, we can exclude the model file and the app will work without predictions (or use a smaller fallback model).
