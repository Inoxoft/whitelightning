#!/bin/bash

# When running with --user flag, the container process runs as the host user
# so files are created with correct permissions automatically
# We only need to switch to appuser when running as root (without --user flag)

if [ "$(id -u)" = "0" ]; then
    # Running as root, switch to appuser
    exec gosu appuser python -m text_classifier.agent "$@"
else
    # Running as non-root user, execute directly
    exec python -m text_classifier.agent "$@"
fi 