#!/bin/bash
SESSION="enrich_session"

tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION "python enrich_roosters.py; exec bash"
echo "Enrich script running in tmux session '$SESSION'."

