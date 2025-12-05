sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-distutils -y

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch huggingface transformers kaggle hf_transfer

export KAGGLE_API_TOKEN=KGAT_47fa0afab6edd8f9a7856ed3ae7a05c8
kaggle competitions download -c llm-classification-finetuning

# --- unzip latest download into ./data ---
latest_zip=$(ls -t *.zip | head -n 1)
unzip "$latest_zip" -d data
rm "$latest_zip"
