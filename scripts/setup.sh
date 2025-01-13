#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "ROOT_DIR: $ROOT_DIR"

# install MT3+
echo "Installing MT3+"
INSTALL_DIR="$ROOT_DIR/dawify/third_party/amt"
if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
    git clone git@github.com:kaiyolau/amt.git "$INSTALL_DIR"
else
    echo "MT3+ already installed"
fi
cd "$INSTALL_DIR/src"
pip install -r requirements.txt

# # install pytorch; demucs and mt3 could break pytorch versions
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install Apollo
echo "Installing Apollo"
APOLLO_INSTALL_DIR="$ROOT_DIR/dawify/third_party/Apollo"
if [ ! -d "$APOLLO_INSTALL_DIR" ]; then
    mkdir -p "$APOLLO_INSTALL_DIR"
    git clone https://github.com/JusperLee/Apollo.git "$APOLLO_INSTALL_DIR"
    cd "$APOLLO_INSTALL_DIR"

    echo "Downloading Apollo models"
    APOLLO_MODEL_DIR="$APOLLO_INSTALL_DIR/model"
    mkdir -p "$APOLLO_MODEL_DIR"

    cd "$APOLLO_MODEL_DIR"
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
else
    echo "Apollo already installed"
fi

################################################################################
# Install dependencies for post processing
################################################################################
cd "$ROOT_DIR"
pip install -U -r requirements.post.txt
