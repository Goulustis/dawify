#!/bin/bash

## THIS INSTALL MT3
INSTALL_DIR="dawify/third_party/amt"
mkdir -p "$INSTALL_DIR"

aws s3 cp s3://amt-deploy-public/amt/ "$INSTALL_DIR" --no-sign-request --recursive
cd "$INSTALL_DIR/src"
python -m pip install -r requirements.txt