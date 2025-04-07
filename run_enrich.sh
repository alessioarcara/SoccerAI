#!/bin/bash
SESSION="enrich_session"
API_REPO="https://github.com/felipeall/transfermarkt-api.git"
API_DIR="data/scraping/transfermarkt-api"
HOST="127.0.0.1"
PORT="8000"

if [ ! -d "$API_DIR" ]; then
    echo "Cloning TransfertMarket API repo..."
    git clone "$API_REPO"
fi 


cd "$API_DIR" || exit 
echo "Installing API dependencies..."

# Start TransfertMarket API server in a tmux session.
tmux kill-session -t "${SESSION}_api" 2>/dev/null
echo "Starting the API server on $HOST:$PORT..."
tmux new-session -d -s "${SESSION}_api" "fastapi run --host $HOST --port $PORT; exec bash"
cd .. 


# Start the enrich_rooster script in a tmux session.
tmux kill-session -t $SESSION 2>/dev/null
echo "Starting enrich_roosters script..."
tmux new-session -d -s $SESSION "python -m data.scraping.enrich_roosters $HOST $PORT; exec bash"
echo "Enrich script running in tmux session '$SESSION'."

