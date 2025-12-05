sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-distutils -y
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install torch
pip install huggingface
pip install kaggle
kaggle competitions download -c llm-classification-finetuning