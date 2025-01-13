#!/bin/bash
set -e

# Run the setup script
echo "Running setup.sh..."
bash /app/scripts/setup.sh

# Execute the main process specified in CMD
echo "Starting main process..."
exec "$@"
