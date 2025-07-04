# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that some Python packages might need
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code into the container
COPY . .

# Expose the port your app runs on
EXPOSE 8050

# Command to run the app using Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:server"]