
apt update
apt install tmux
apt install vim

source .venv/bin/activate
pkill -f "torchrun"
sleep 1
torchrun --nproc_per_node=4 python run_5_models.py --base_model roberta-base --tag roberta
pkill -f "torchrun"
sleep 1
torchrun --nproc_per_node=4 python run_5_models.py --base_model google/electra-base-discriminator --tag electra
pkill -f "torchrun"
sleep 1
torchrun --nproc_per_node=4 python run_5_models.py --base_model microsoft/deberta-base --tag deberta
pkill -f "torchrun"
sleep 1
torchrun --nproc_per_node=4 python run_5_models.py --base_model albert-base-v2 --tag albert