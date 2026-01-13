"""
Vercel serverless function entry point for Dash app.
Vercel's Python runtime automatically handles WSGI apps.
"""

import os
import sys

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the Dash app
from app import app

# Vercel's Python runtime expects a WSGI application
# Dash apps are built on Flask, so we expose the underlying Flask server
# Vercel will automatically handle the WSGI interface
application = app.server
