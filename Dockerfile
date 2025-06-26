# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file
COPY requirements/base.txt .

# Install dependencies
RUN pip install --no-cache-dir -r base.txt

# Copy the application code
COPY text_classifier/ ./text_classifier/

# Create own_data directory for user datasets
RUN mkdir -p /app/own_data

# Set environment variable to ensure Python output is not buffered
ENV PYTHONUNBUFFERED=1

# Command to run the CLI
# Users will override the prompt via Docker run command
ENTRYPOINT ["python", "-m", "text_classifier.agent"]

CMD []