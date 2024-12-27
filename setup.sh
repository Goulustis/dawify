#!/bin/bash

# install MT3; DO NOT CHANGE INSTALL_DIR!!
INSTALL_DIR="dawify/third_party/amt"
mkdir -p "$INSTALL_DIR"

aws s3 cp s3://amt-deploy-public/amt/ "$INSTALL_DIR" --no-sign-request --recursive
cd "$INSTALL_DIR/src"
python -m pip install -r requirements.txt

# install pytorch; demucs and mt3 could break pytorch versions
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# download FluidR3_GM soundfont; required for midi -> audio conversion
ASSETS_DIR="./assets"
mkdir -p "$ASSETS_DIR"
wget -P "$ASSETS_DIR" https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip
unzip "$ASSETS_DIR/FluidR3_GM.zip" -d "$ASSETS_DIR"