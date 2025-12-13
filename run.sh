source .venv/bin/activate
torchrun --nproc_per_node=1 run_4_size.py --base_model bert-base-uncased