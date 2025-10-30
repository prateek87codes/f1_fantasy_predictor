# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that some Python packages might need
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code into the container
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Command to run the app using a tuned Gunicorn server for Cloud Run
# - gthread worker class with multiple threads improves concurrency for I/O-bound Dash callbacks
# - preload reduces per-worker import time; ensure imports do not perform heavy network calls
# - increased timeout helps first-time cache fills
CMD ["gunicorn", "-k", "gthread", "--threads", "4", "--workers", "2", "--timeout", "180", "--preload", "--bind", "0.0.0.0:8080", "wsgi:application"]
