# gunicorn_config.py
# This tells Gunicorn how to run your Dash app.
# The --preload flag is important for caching to work correctly across multiple workers.
# You can adjust the number of workers if needed.

workers = 2
bind = "0.0.0.0:8050"
preload_app = True