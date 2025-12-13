#!/bin/bash

# Create timestamped session name
SESSION="train_$(date +%Y%m%d_%H%M%S)"

# Clean up any existing torchrun processes
pkill -f "torchrun"
lsof -ti:29500 | xargs kill -9 2>/dev/null
sleep 1

# Create new tmux session, run training, then detach
tmux new-session -d -s "$SESSION" "source .venv/bin/activate && torchrun --nproc_per_node=4 run_4_size.py --base_model bert-large-uncased"

echo "Training started in tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"