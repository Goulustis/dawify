# audio_pre_processing.py

import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import Type
import glob
import os.path as osp
import os

from dawify.dw_config import InstantiateConfig
from dataclasses import dataclass, field

@dataclass
class AudioPreProcessorConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: AudioPreProcessor)

    # --- Equalizer parameters ---
    freq: int = 8000
    gain_db: float = 0.0
    q_factor: float = 2
    
    # --- Expander parameters ---
    threshold: float = 0.05
    ratio: float = 1.0

    # --- Compressor parameters (for guitar) ---
    compressor_threshold: float = 0.1
    compressor_ratio: float = 3.5
    compressor_output_gain: float = 1.1

    # --- Noise Gate parameters (for guitar) ---
    gate_threshold: float = 0.005
    gate_reduction: float = 0.4

    # --- Track-specific Gain (in dB) ---
    drums_gain_db: float = 0.0
    guitar_gain_db: float = 0.0
    bass_gain_db:   float = 0.0
    piano_gain_db:  float = 0.0

    def __post_init__(self):
        # Original params
        self.eq_params = {
            'freq': self.freq,
            'gain_db': self.gain_db,
            'q_factor': self.q_factor
        }
        self.expander_params = {
            'threshold': self.threshold,
            'ratio': self.ratio
        }
        
        # New params
        self.compressor_params = {
            'threshold': self.compressor_threshold,
            'ratio': self.compressor_ratio,
            'output_gain': self.compressor_output_gain
        }
        self.noise_gate_params = {
            'threshold': self.gate_threshold,
            'reduction': self.gate_reduction
        }

class AudioPreProcessor:
    def __init__(self, config: AudioPreProcessorConfig):
        """
        Initialize the AudioPreProcessor with various parameters.
        """
        self.config = config
        # Existing
        self.eq_params = config.eq_params
        self.expander_params = config.expander_params
        # New
        self.compressor_params = config.compressor_params
        self.noise_gate_params = config.noise_gate_params

        # Track gain in dB
        self.drums_gain_db  = config.drums_gain_db
        self.guitar_gain_db = config.guitar_gain_db
        self.bass_gain_db   = config.bass_gain_db
        self.piano_gain_db  = config.piano_gain_db

        self.out_dir = "outputs/preprocessed"

        os.makedirs(self.out_dir, exist_ok=True)

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalizes audio to ensure it stays within the range [-1, 1].
        """
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 1:
            audio_data = audio_data / max_amplitude
        return audio_data
    
    def apply_limiter(self, audio_data, limit=1.0):
        """
        A very basic limiter that clips samples outside [-limit, limit].
        """
        return np.clip(audio_data, -limit, limit)


    def apply_gain(self, audio_data: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Apply gain (in dB) to the audio data and normalize to prevent clipping.
        """
        linear_gain = 10 ** (gain_db / 20.0)
        audio_data = audio_data * linear_gain
        return audio_data


    def apply_equalizer(self, audio_data, sample_rate):
        """
        Apply a parametric equalizer to boost the target frequency and normalize to prevent clipping.
        """
        b, a = signal.iirpeak(
            self.eq_params['freq'],
            self.eq_params.get('q_factor', 2),
            fs=sample_rate
        )
        filtered_audio = signal.lfilter(b, a, audio_data)
        gain_linear = 10 ** (self.eq_params['gain_db'] / 20)
        equalized_audio = filtered_audio * gain_linear
        combined_audio = (audio_data + equalized_audio) / 2
        return combined_audio


    def apply_expander(self, audio_data):
        """
        Apply a simple expander to restore dynamics and normalize to prevent clipping.
        """
        threshold = self.expander_params['threshold']
        ratio = self.expander_params['ratio']

        def expander(x):
            return x if abs(x) > threshold else x / ratio

        expanded_audio = np.vectorize(expander)(audio_data)
        return expanded_audio


    def apply_compressor(self, audio_data):
        """
        Apply a simple compressor and normalize to prevent clipping.
        """
        threshold = self.compressor_params['threshold']
        ratio = self.compressor_params['ratio']
        output_gain = self.compressor_params['output_gain']

        def compress_sample(x):
            amp = abs(x)
            sign = np.sign(x)
            if amp > threshold:
                compressed = threshold + (amp - threshold) / ratio
                return sign * compressed
            else:
                return x

        compressed_audio = np.vectorize(compress_sample)(audio_data)
        compressed_audio *= output_gain
        return compressed_audio


    def apply_noise_gate(self, audio_data):
        """
        Apply a simple noise gate and normalize to prevent clipping.
        """
        gate_threshold = self.noise_gate_params['threshold']
        reduction = self.noise_gate_params['reduction']

        def gate_sample(x):
            return x * reduction if abs(x) < gate_threshold else x

        gated_audio = np.vectorize(gate_sample)(audio_data)
        return gated_audio
    
    def apply_low_cut_filter(self, audio_data, sample_rate, cutoff=500.0):
        """
        Apply a low-cut (high-pass) filter to remove frequencies below the cutoff.
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
        filtered_audio = signal.lfilter(b, a, audio_data)
        return filtered_audio


    # ------------------------------------------------------------------
    #                Processors for Each Instrument
    # ------------------------------------------------------------------
    def process_drum_files(self, separated_files):
        """
        Drums chain: expander -> equalizer -> gain -> limiter
        (No normalization)
        """
        processed_files = []
        for audio_file in separated_files:
            if "drums.wav" in audio_file:
                audio, sr = sf.read(audio_file)
                # Do NOT sum to mono; keep as-is
                # (Assume shape is (num_samples,) for mono or (num_samples, channels) for stereo)

                # 3) Adjust gain
                audio = self.apply_gain(audio, self.drums_gain_db)
                # 1) Expander
                audio = self.apply_expander(audio)
                # 2) Equalizer
                audio = self.apply_equalizer(audio, sr)
                # 4) Limiter
                audio = self.apply_limiter(audio)

                output_file = osp.join(self.curr_save_dir, osp.basename(audio_file))
                sf.write(output_file, audio, sr)
                processed_files.append(output_file)
            else:
                processed_files.append(audio_file)
        return processed_files

    def process_guitar_files(self, separated_files):
        """
        Guitar chain: low-cut filter -> compressor -> noise gate -> gain -> limiter
        (No normalization)
        """
        processed_files = []
        for audio_file in separated_files:
            if "guitar.wav" in audio_file:
                audio, sr = sf.read(audio_file)

                # 1) Low-Cut Filter
                audio = self.apply_low_cut_filter(audio, sr)
                # 2) Compressor
                # audio = self.apply_compressor(audio)
                # 3) Adjust gain
                audio = self.apply_gain(audio, self.guitar_gain_db)
                # 4) Limiter
                audio = self.apply_limiter(audio)

                output_file = osp.join(self.curr_save_dir, osp.basename(audio_file))
                sf.write(output_file, audio, sr)
                processed_files.append(output_file)
            else:
                processed_files.append(audio_file)
        return processed_files

    def process_bass_files(self, separated_files):
        """
        Bass chain: expander -> gain -> limiter
        (No normalization)
        """
        processed_files = []
        for audio_file in separated_files:
            if "bass.wav" in audio_file:
                audio, sr = sf.read(audio_file)

                # 2) Adjust gain
                audio = self.apply_gain(audio, self.bass_gain_db)
                # 1) Expander
                audio = self.apply_expander(audio)
                # 3) Limiter
                audio = self.apply_limiter(audio)

                output_file = osp.join(self.curr_save_dir, osp.basename(audio_file))
                sf.write(output_file, audio, sr)
                processed_files.append(output_file)
            else:
                processed_files.append(audio_file)
        return processed_files

    def process_piano_files(self, separated_files):
        """
        Piano chain: low-cut filter -> expander -> gain -> limiter
        (No normalization)
        """
        processed_files = []
        for audio_file in separated_files:
            if "piano.wav" in audio_file:
                audio, sr = sf.read(audio_file)

                # 1) Low-Cut Filter
                audio = self.apply_low_cut_filter(audio, sr)
                # 2) Expander
                audio = self.apply_expander(audio)
                # 3) Adjust gain
                audio = self.apply_gain(audio, self.piano_gain_db)
                # 4) Limiter
                audio = self.apply_limiter(audio)

                output_file = osp.join(self.curr_save_dir, osp.basename(audio_file))
                sf.write(output_file, audio, sr)
                processed_files.append(output_file)
            else:
                processed_files.append(audio_file)
        return processed_files

    # ------------------------------------------------------------------
    #                Example "Process All" Method
    # ------------------------------------------------------------------
    def process(self, separated_files):
        """
        Example logic to route each separated file to the correct 
        instrument chain. Bypasses normalization.
        """

        self.curr_save_dir = osp.join(self.out_dir, osp.basename(osp.dirname(separated_files[0])))
        os.makedirs(self.curr_save_dir, exist_ok=True)

        final_processed = []
        for fpath in separated_files:
            if "drums.wav" in fpath:
                proc_files = self.process_drum_files([fpath])
                final_processed.extend(proc_files)
            elif "piano.wav" in fpath:
                proc_files = self.process_piano_files([fpath])
                final_processed.extend(proc_files)
            elif "guitar.wav" in fpath:
                proc_files = self.process_guitar_files([fpath])
                final_processed.extend(proc_files)
            elif "bass.wav" in fpath:
                proc_files = self.process_bass_files([fpath])
                final_processed.extend(proc_files)
            else:
                # No special processing
                final_processed.append(fpath)
        return final_processed

    def get_out_fs(self):
        if self.curr_save_dir is None:
            return []
        return sorted(glob.glob(osp.join(self.curr_save_dir, "*.wav")))
