"""
Main entry point for Google App Engine deployment.
This file imports the Dash app from app.py and makes it available to App Engine.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Dash app
from app import app

# For Google App Engine, we need to expose the WSGI application
# Dash apps are built on Flask, so we can access the underlying Flask app
# App Engine will look for a variable named 'app' or 'application'
application = app.server

# Also expose as 'app' for compatibility
app_wsgi = app.server

if __name__ == '__main__':
    # This won't be used in production (App Engine handles this)
    # But useful for local testing
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
