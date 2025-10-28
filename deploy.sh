#!/bin/bash

# F1 Fantasy Predictor - Google Cloud Deployment Script
# This script helps deploy the application to Google App Engine

echo "🏎️  F1 Fantasy Predictor - Google Cloud Deployment"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK is not installed."
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Set default project (you'll need to replace this with your project ID)
echo "📋 Setting up Google Cloud project..."
echo "Please enter your Google Cloud Project ID:"
read -p "Project ID: " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Project ID is required"
    exit 1
fi

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Deploy the application
echo "🚀 Deploying to Google App Engine..."
gcloud app deploy app.yaml --quiet

# Get the URL
echo "✅ Deployment complete!"
echo "🌐 Your application is available at:"
gcloud app browse

echo ""
echo "📝 Next steps:"
echo "1. Set your OpenWeather API key in the Google Cloud Console"
echo "2. Go to App Engine > Settings > Environment Variables"
echo "3. Add OPENWEATHER_API_KEY with your actual API key"
echo "4. Redeploy if needed: gcloud app deploy app.yaml"
