#!/bin/bash

ROOT_DIR=$(pwd)

# install MT3+; DO NOT CHANGE INSTALL_DIR!!
echo "Installing MT3+"
INSTALL_DIR="dawify/third_party/amt"
mkdir -p "$INSTALL_DIR"

aws s3 cp s3://amt-deploy-public/amt/ "$INSTALL_DIR" --no-sign-request --recursive
cd "$INSTALL_DIR/src"
python -m pip install -r requirements.txt

# install pytorch; demucs and mt3 could break pytorch versions
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install Apollo
echo "Installing Apollo"
APOLLO_INSTALL_DIR="dawify/third_party/Apollo"
APOLLO_INSTALL_DIR=$(realpath "$APOLLO_INSTALL_DIR")
git clone https://github.com/JusperLee/Apollo.git "$APOLLO_INSTALL_DIR"
cd "$APOLLO_INSTALL_DIR"

mkdir model
cd model
wget 'https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin'
wget 'https://huggingface.co/jarredou/lew_apollo_vocal_enhancer/resolve/main/apollo_model.ckpt'
wget 'https://huggingface.co/jarredou/lew_apollo_vocal_enhancer/resolve/main/apollo_model_v2.ckpt'
wget 'https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/apollo_model_uni.ckpt'

cd "$APOLLO_INSTALL_DIR/configs"
wget 'https://huggingface.co/jarredou/lew_apollo_vocal_enhancer/resolve/main/config_apollo_vocal.yaml'
wget 'https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/config_apollo_uni.yaml'

cd "$APOLLO_INSTALL_DIR"
rm  inference.py
wget 'https://raw.githubusercontent.com/jarredou/Apollo-Colab-Inference/main/inference.py'