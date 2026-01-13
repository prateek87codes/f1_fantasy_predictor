"""
Vercel serverless function entry point for Dash app.
Vercel's Python runtime automatically handles WSGI apps.
"""

import os
import sys

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the Dash app's Flask server
# Vercel expects the WSGI app to be named 'app'
from app import server as app
