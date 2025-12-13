#!/bin/bash

SESSION="train_$(date +%Y%m%d_%H%M%S)"

pkill -f "torchrun"
lsof -ti:29500 | xargs kill -9 2>/dev/null
sleep 1

# NCCL tuning for NUMA systems
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # Disable InfiniBand (you don't have NVLink)
export NCCL_P2P_LEVEL=SYS         # Allow cross-NUMA communication
export NCCL_SOCKET_IFNAME=lo      # Use loopback for single-node
export NCCL_ASYNC_ERROR_HANDLING=1

tmux new-session -d -s "$SESSION" "source .venv/bin/activate && torchrun --nproc_per_node=4 run_4_size.py --base_model bert-large-uncased"

echo "Training started in tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"