#!/bin/bash

## THIS INSTALL MT3
INSTALL_DIR="dawify/third_party/amt"

aws s3 cp s3://amt-deploy-public/amt/ "$INSTALL_DIR" --no-sign-request --recursive
cd "$INSTALL_DIR"
python -m pip install -r requirements.txt