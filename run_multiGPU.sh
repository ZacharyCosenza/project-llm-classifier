#!/bin/bash

apt update
apt install tmux

SESSION="train_$(date +%Y%m%d_%H%M%S)"

pkill -f "torchrun"
sleep 1

tmux new-session -d -s "$SESSION" "source .venv/bin/activate && torchrun --nproc_per_node=4 run_4_size.py --base_model bert-base-uncased"

echo "Training started in tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"