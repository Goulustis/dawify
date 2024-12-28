import os
import torch
import librosa
import soundfile as sf
from tqdm.auto import tqdm
import numpy as np
import yaml
from ml_collections import ConfigDict
import sys
import os.path as osp

from torch import nn


apollo_dir = osp.abspath(osp.join(osp.dirname(__file__), "..", "third_party", "Apollo"))
if apollo_dir not in sys.path:
    sys.path.append(apollo_dir)

import look2hear.models
import warnings
warnings.filterwarnings("ignore")

def get_config(config_path):
    with open(config_path) as f:
        #config = OmegaConf.load(config_path)
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=False, sr=44100)
    print(f'INPUT audio.shape = {audio.shape} | samplerate = {samplerate}')
    #audio = dBgain(audio, -6)
    return torch.from_numpy(audio), samplerate

def save_audio(file_path, audio, samplerate=44100):
    #audio = dBgain(audio, +6)
    sf.write(file_path, audio.T, samplerate, subtype="PCM_16")

def process_chunk(chunk, model):
    chunk = chunk.unsqueeze(0).cuda()
    with torch.no_grad():
        return model(chunk).squeeze(0).squeeze(0).cpu()

def _getWindowingArray(window_size, fade_size):
    # IMPORTANT NOTE :
    # no fades here in the end, only removing the failed ending of the chunk
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def dBgain(audio, volume_gain_dB):
    gain = 10 ** (volume_gain_dB / 20)
    gained_audio = audio * gain 
    return gained_audio


def inference(input_wav, output_wav, model, chunk_size=10, overlap=2):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    test_data, samplerate = load_audio(input_wav)
    
    C = chunk_size * samplerate  # chunk_size seconds to samples
    N = overlap
    step = C // N
    fade_size = 3 * 44100 # 3 seconds
    print(f"N = {N} | C = {C} | step = {step} | fade_size = {fade_size}")
    
    border = C - step
    
    # handle mono inputs correctly
    if len(test_data.shape) == 1:
        test_data = test_data.unsqueeze(0) 

    # Pad the input if necessary
    if test_data.shape[1] > 2 * border and (border > 0):
        test_data = torch.nn.functional.pad(test_data, (border, border), mode='reflect')

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)

    i = 0
    progress_bar = tqdm(total=test_data.shape[1], desc="Processing audio chunks", leave=False)

    while i < test_data.shape[1]:
        part = test_data[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
            else:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

        out = process_chunk(part, model)

        window = windowingArray
        if i == 0:  # First audio chunk, no fadein
            window[:fade_size] = 1
        elif i + C >= test_data.shape[1]:  # Last audio chunk, no fadeout
            window[-fade_size:] = 1

        result[..., i:i+length] += out[..., :length] * window[..., :length]
        counter[..., i:i+length] += window[..., :length]

        i += step
        progress_bar.update(step)

    progress_bar.close()

    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    # Remove padding if added earlier
    if test_data.shape[1] > 2 * border and (border > 0):
        final_output = final_output[..., border:-border]

    save_audio(output_wav, final_output, samplerate)
    print(f'Success! Output file saved as {output_wav}')

    # Memory clearing
    # model.cpu()
    # del model
    torch.cuda.empty_cache()

# NOTE: Details here: https://github.com/jarredou/Apollo-Colab-Inference/blob/main/Apollo_Audio_Restoration_Colab.ipynb
MODEL_PATH = {
    # Lew Universal Lossy Enhancer
    "apollo_uni": osp.join(apollo_dir, "model", "apollo_model_uni.ckpt"),
    # Lew Vocal Enhancer v2(beta)
    "apollo_v2" : osp.join(apollo_dir, "model", "apollo_model_v2.ckpt"),
    # mp3 enhancer
    "apollo" : osp.join(apollo_dir, "model", "apollo_model.ckpt"),
    # MP3 Enhancer
    "pytorch" : osp.join(apollo_dir, "model", "pytorch_model.bin")
}

MODEL_CONFIG = {
    "pytorch" : osp.join(apollo_dir, "configs", "apollo.yaml"),
    "apollo" : osp.join(apollo_dir, "configs", "apollo.yaml"),
    "apollo_v2" : osp.join(apollo_dir, "configs", "config_apollo_vocal.yaml"),
    "apollo_uni" : osp.join(apollo_dir, "configs", "config_apollo_uni.yaml")
}


def load_model(model_name:str):
    ckpt_path = MODEL_PATH[model_name]
    config_path = MODEL_CONFIG[model_name]
    config = get_config(config_path)

    feature_dim = config['model']['feature_dim']
    sr = config['model']['sr']
    win = config['model']['win']
    layer = config['model']['layer']

    return look2hear.models.BaseModel.from_pretrain(ckpt_path, sr=sr, win=win, feature_dim=feature_dim, layer=layer).cuda()