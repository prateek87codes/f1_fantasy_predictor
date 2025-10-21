#!/bin/bash

# F1 Fantasy Predictor Startup Script
echo "🏎️  Starting F1 Fantasy Predictor..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python -c "import dash, fastf1, pandas, numpy, plotly, xgboost, sklearn, joblib, requests, tqdm, pytz" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Dependencies not installed. Installing now..."
    pip install -r requirements.txt
fi

# Start the application
echo "🚀 Starting F1 Fantasy Predictor..."
echo "   The app will be available at: http://127.0.0.1:8050"
echo "   Press Ctrl+C to stop the application"
echo ""

python app.py
