FROM python:3.11-slim

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

# Copy the application code
COPY text_classifier/ ./text_classifier/

# Create directories for models and data
# Note: When you mount volumes, the permissions will be determined by the host.
RUN mkdir -p /app/own_data /app/models

# Switch to the non-root user
USER appuser

# Set the default command
CMD ["python", "-m", "text_classifier.agent"]
