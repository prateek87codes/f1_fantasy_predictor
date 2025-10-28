# F1 Fantasy Predictor - Deployment Guide

This document outlines the process and configuration used to deploy the F1 Fantasy Predictor application to Google Cloud Run.

## Deployment Platform: Google Cloud Run

After initial attempts with Google App Engine, the project was deployed to **Google Cloud Run**. Cloud Run was chosen for its flexibility in handling containerized applications, which is a better fit for the memory and startup requirements of this data-intensive Dash application.

## Key Configuration Files

Several files were added to the repository to enable deployment:

-   `Dockerfile`: This file contains the instructions to build a Docker container for the application. It specifies the Python version, installs dependencies from `requirements.txt`, and sets the command to run the application using the `gunicorn` web server.

-   `wsgi.py`: A simple WSGI (Web Server Gateway Interface) entry point. It imports the main `app` object from `app.py` and exposes its server attribute, which `gunicorn` uses to run the application.

-   `.gcloudignore`: Similar to a `.gitignore` file, this tells the Google Cloud SDK which files and directories to exclude from the deployment to keep the container light and secure.

## Deployment Strategy: Local Docker Build

Due to persistent issues with the Google Cloud Build service, a local build strategy was adopted:

1.  **Build Locally:** The Docker container image was built on the local machine, specifying the `--platform linux/amd64` flag to ensure compatibility with the Cloud Run environment (which uses an AMD64 architecture).
2.  **Push to Artifact Registry:** The locally built image was pushed to a private Google Artifact Registry repository.
3.  **Deploy from Registry:** The pre-built image was then deployed to Cloud Run.

This approach bypasses the cloud-based build process, providing more control and avoiding potential platform-specific caching issues.

## Cloud Run Service Configuration

The service was configured with the following key settings to ensure performance:

-   **Memory:** `4Gi` - Increased from the default to accommodate the loading of large `.csv` data files and `.joblib` machine learning models.
-   **CPU:** `2` - Increased from the default to provide more processing power, especially during the application's startup phase.
-   **Minimum Instances:** `1` - Set to keep at least one container instance "warm" at all times. This eliminates "cold starts" and ensures the application is always responsive, providing a much better user experience at the cost of a small, continuous resource usage.

## API Key Management

The `PERPLEXITY_API_KEY` is managed securely using **Google Secret Manager**:

1.  A secret was created in Secret Manager to hold the API key.
2.  The Cloud Run service was granted the `Secret Manager Secret Accessor` IAM role to allow it to read the secret.
3.  The secret was mounted into the Cloud Run service as an environment variable named `PERPLEXITY_API_KEY`, which the application code (`app.py`) reads at runtime.
