FROM tensorflow/tensorflow:2.11.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements/base.txt requirements/base.txt
COPY requirements/dev.txt requirements/dev.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/base.txt

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "text_classifier.agent"] 