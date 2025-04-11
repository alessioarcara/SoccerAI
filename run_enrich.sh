#!/bin/bash
# Set variables
SESSION="enrich_session"
API_REPO="https://github.com/felipeall/transfermarkt-api.git"
API_DIR="transfermarkt-api"
HOST="localhost"
PORT="8000"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Check if the API repository exists in the expected location.
if [ ! -d "$PROJECT_ROOT/data/scraping/$API_DIR" ]; then
    echo "Cloning Transfermarkt API repo..."
    cd "$PROJECT_ROOT/data/scraping" || exit
    git clone "$API_REPO"
fi 

# Install API dependencies.
cd "$PROJECT_ROOT/data/scraping/$API_DIR" || exit
echo "Installing API dependencies..."
pip install -r requirements.txt

# Start the API server in a detached tmux session, setting the working directory.
tmux kill-session -t "${SESSION}_api" 2>/dev/null
echo "Starting the API server on $HOST:$PORT..."
tmux new-session -d -s "${SESSION}_api" -c "$(pwd)" "uvicorn app.main:app --host $HOST --port $PORT; exec bash"

# --- Return to Project Root ---
cd "$PROJECT_ROOT" || exit
echo "Project root: $(pwd)"

# Start the enrich_roosters script in another detached tmux session,
tmux kill-session -t "$SESSION" 2>/dev/null
echo "Starting enrich_roosters script..."
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "python -m data.scraping.enrich_roosters --host $HOST --port $PORT; exec bash"
echo "Enrich script running in tmux session '$SESSION'."
