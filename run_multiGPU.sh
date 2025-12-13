source .venv/bin/activate
torchrun --nproc_per_node=2 run_4_size.py --base_model bert-base-uncased --tag multigpu