# download local 
# - https://github.com/facebookresearch/demucs/issues/367
# - https://github.com/facebookresearch/demucs/issues/391

# saved path:
# $TORCH_HOME/hub, if environment variable TORCH_HOME is set.
# $XDG_CACHE_HOME/torch/hub, if environment variable XDG_CACHE_HOME is set.
# ~/.cache/torch/hub

MODEL_DIR=pretrain_ckpts/demucs
# for list of models see: https://github.com/facebookresearch/demucs/tree/main/demucs/remote
MODEL_HASH=955717e8
INPUT_MP3="assets/All the Way North.mp3"
# INPUT_MP3="assets/north_20sec.mp3"
# INPUT_MP3="/home/boss/projects/dawify/assets/for-her-chill-upbeat-summel-travel-vlog-and-ig-music-royalty-free-use-202298.mp3"



# 1. download the model
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
fi

MODEL_FILE="$MODEL_DIR/$MODEL_HASH.th"
if [ ! -f "$MODEL_FILE" ]; then
    wget -O "$MODEL_FILE" "https://dl.fbaipublicfiles.com/demucs/mdx_final/$MODEL_HASH.th"
    # python download_demucs_models.py "https://dl.fbaipublicfiles.com/demucs/mdx_final/$MODEL_HASH.th" "$MODEL_FILE"
else
    echo "Model already exists. No need for download."
fi

# run local -- not working
# python -m demucs -s 955717e8 --repo $MODEL_DIR --mp3 $INPUT_MP4 --out outputs/seperated

# run by downloading the model
# python -m demucs -s 955717e8 --mp3 "$INPUT_MP4" --out outputs/seperated
# python -m demucs -n htdemucs --mp3 "$INPUT_MP3" --out outputs/seperated
demucs "$INPUT_MP3" -n htdemucs -j 4 --out outputs/seperated
