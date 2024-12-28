import numpy as np
from scipy.io import wavfile
import subprocess
import rich
from rich.table import Table

import os.path as osp
import glob

CONSOLE = rich.get_console()

def midi_to_wav_timidity(midi_path, output_wav="tmp.wav", sample_rate=44100):
    command = [
        'timidity',
        midi_path,
        '-Ow',                # Output as WAV
        '-o', output_wav,     # Specify output file
        '-s', str(sample_rate)  # Sample rate
    ]
    subprocess.run(command, check=True)
    return output_wav


def calculate_snr(original_wav, midi_path):
    # Read audio files
    rate1, original = wavfile.read(original_wav)
    recon_wav_f = midi_to_wav_timidity(midi_path)
    rate2, reconstructed = wavfile.read(recon_wav_f) #midi_to_wav_timidity(midi_path)

    # Ensure they have the same sampling rate
    assert rate1 == rate2, "Sampling rates do not match"
    
    # Truncate to the shortest length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Normalize signals
    original = original / np.max(np.abs(original))
    reconstructed = reconstructed / np.max(np.abs(reconstructed))

    # Calculate the SNR
    noise = original - reconstructed
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))
    
    return snr


def print_metrics(metrics: dict):
    table = Table(title="Backtest Metrics")

    table.add_column("Music", justify="right", style="cyan", no_wrap=True)
    table.add_column("Signal Noise Ratio", style="magenta")

    for metric, value in metrics.items():
        table.add_row(metric, f"{value:.4f}")

    CONSOLE.print(table)

def calc_and_print_snrs(recon_dir, original_dir):
    recon_fs = sorted(glob.glob(osp.join(recon_dir, "*.mid")))

    snrs = {}
    for f in recon_fs:
        snr = calculate_snr(osp.join(original_dir, osp.basename(f).replace(".mid", ".wav")), f)
        snrs[osp.basename(f)] = snr
    
    print_metrics(snrs)

if __name__ == "__main__":
    ori = "outputs/demuc/htdemucs_6s/full_tracks/bass.wav"
    recon = "outputs/mt3/full_tracks/bass.mid"
    # print(calculate_snr(ori, recon))
    calc_and_print_snrs(osp.dirname(recon), osp.dirname(ori))