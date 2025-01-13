FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install the package with dependencies
RUN pip install --upgrade pip
RUN python -m pip install -e .

# Verify the installation
RUN python -c "import demucs; import tyro"

ENTRYPOINT ["bash", "/app/scripts/entrypoint.sh"]
