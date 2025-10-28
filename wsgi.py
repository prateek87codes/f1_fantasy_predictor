"""
WSGI entry point for Google Cloud Run.
This file provides the WSGI application that Cloud Run will use.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Dash app
from app import app

# Cloud Run expects a WSGI application
# Dash apps are built on Flask, so we can access the underlying Flask app
application = app.server

if __name__ == '__main__':
    # For local testing
    port = int(os.environ.get('PORT', 8080))
    application.run(host='0.0.0.0', port=port, debug=False)
