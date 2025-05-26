#!/bin/bash
set -e # Exit on error

# Configuration variables
SESSION_NAME="enrich_session"
API_REPO="https://github.com/felipeall/transfermarkt-api.git"
API_DIR="transfermarkt-api"
API_HOST="localhost"
API_PORT="8000"

# Path definitions
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PATH="${PROJECT_ROOT}/${API_DIR}"

echo "======================================"
echo "Starting Transfermarkt data enrichment process"
echo "Project directory: $PROJECT_ROOT"
echo "======================================"

# Check if API repository exists
if [ ! -d "$API_PATH" ]; then
    echo "Cloning Transfermarkt API repository..."
    git clone "$API_REPO"
fi 

# Install dependencies
echo "Installing dependencies..."
pip install -e .
cd "$API_PATH"
pip install -r requirements.txt -q

# Stop existing tmux sessions
echo "Cleaning up existing tmux sessions..."
tmux kill-session -t "${SESSION_NAME}_api" 2>/dev/null || true
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Start API server in a tmux session
echo "Starting API server on $API_HOST:$API_PORT..."
cd "$API_PATH"
tmux new-session -d -s "${SESSION_NAME}_api" "uvicorn app.main:app --host $API_HOST --port $API_PORT; exec bash"

# Wait for API server to start up
echo "Waiting for API server to initialize..."
sleep 3

# Start enrichment script in another tmux session
echo "Starting enrich_roosters script..."
cd "$PROJECT_ROOT"
tmux new-session -d -s "$SESSION_NAME" "python ./scripts/enrich_rosters.py --host $API_HOST --port $API_PORT; exec bash"

echo "======================================"
echo "Process started successfully!"
echo "To view API logs: tmux attach-session -t ${SESSION_NAME}_api"
echo "To view script logs: tmux attach-session -t ${SESSION_NAME}"
echo "======================================"