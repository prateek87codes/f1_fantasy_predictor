# gunicorn_config.py
workers = 2  # Keep this at 2 for the free tier
bind = "0.0.0.0:8050"
preload_app = True # This is important for caching
timeout = 120 # Increase timeout to allow for slow initial data loads