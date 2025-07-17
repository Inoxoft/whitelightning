#!/bin/bash

# Install gosu if it's not already installed
if ! command -v gosu &> /dev/null
then
    echo "Installing gosu..."
    apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*
fi

# Set permissions for /app/models and /app/own_data to be writable by appuser
chown -R appuser:appuser /app/models /app/own_data
chmod -R ug+rwx /app/models /app/own_data

# Execute the main command as the 'appuser'
exec gosu appuser python -m text_classifier.agent "$@" 