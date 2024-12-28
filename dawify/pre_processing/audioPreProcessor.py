# audio_pre_processing.py
import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import Type
import glob
import os.path as osp

from dawify.dw_config import InstantiateConfig
from dataclasses import dataclass, field

@dataclass
class AudioPreProcessorConfig(InstantiateConfig):
    _target:Type = field(default_factory=lambda: AudioPreProcessor)

    # Add equalizer parameters
    freq: int = 8000
    gain_db: float = 4
    q_factor: float = 2
    
    # Add expander parameters
    threshold: float = 0.05
    ratio: float = 1.0

    def __post_init__(self):
        self.eq_params = {'freq': self.freq, 'gain_db': self.gain_db, 'q_factor': self.q_factor}
        self.expander_params = {'threshold': self.threshold, 'ratio': self.ratio}


class AudioPreProcessor:
    def __init__(self, config: AudioPreProcessorConfig):
        """
        Initialize the AudioPreProcessor with equalizer and expander parameters.

        :param eq_params: Dictionary containing equalizer parameters (e.g., {'freq': 8000, 'gain_db': 6, 'q_factor': 2})
        :param expander_params: Dictionary containing expander parameters (e.g., {'threshold': 0.05, 'ratio': 2})
        """
        self.config = config
        self.eq_params = config.eq_params
        self.expander_params = config.expander_params
        self.curr_save_dir = None

    def apply_equalizer(self, audio_data, sample_rate):
        """
        Apply a parametric equalizer to boost the target frequency.
        """
        b, a = signal.iirpeak(self.eq_params['freq'], self.eq_params.get('q_factor', 2), fs=sample_rate)
        filtered_audio = signal.lfilter(b, a, audio_data)
        gain_linear = 10 ** (self.eq_params['gain_db'] / 20)
        equalized_audio = filtered_audio * gain_linear
        combined_audio = (audio_data + equalized_audio) / 2
        return combined_audio

    def apply_expander(self, audio_data):
        """
        Apply a simple expander to restore dynamics.
        """
        threshold = self.expander_params['threshold']
        ratio = self.expander_params['ratio']

        def expander(x):
            return x if abs(x) > threshold else x / ratio

        expanded_audio = np.vectorize(expander)(audio_data)
        max_amplitude = np.max(np.abs(expanded_audio))
        if max_amplitude > 1:
            expanded_audio = expanded_audio / max_amplitude  # Prevent clipping in expander
        return expanded_audio

    def normalize_audio(self, audio):
        """
        Normalize the audio to ensure it fits within the [-1, 1] range, maintaining RMS.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        desired_rms = 0.1  # Adjust based on desired loudness
        normalization_factor = desired_rms / (rms + 1e-9)  # Avoid division by zero
        normalized_audio = audio * normalization_factor

        max_amplitude = np.max(np.abs(normalized_audio))
        if max_amplitude > 1:
            normalized_audio = normalized_audio / max_amplitude

        return normalized_audio

    def process_audio(self, audio_data, sample_rate):
        """
        Process the audio data by applying an expander, equalizer, and normalization.
        """
        audio_data = self.apply_expander(audio_data)
        audio_data = self.apply_equalizer(audio_data, sample_rate)
        audio_data = self.normalize_audio(audio_data)
        return audio_data

    def process_file(self, input_file, output_file):
        """
        Process the input audio file and save the processed output to a file.
        """
        audio, sample_rate = sf.read(input_file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono if stereo

        processed_audio = self.process_audio(audio, sample_rate)

        sf.write(output_file, processed_audio, sample_rate)
        return output_file

    def process_drum_files(self, separated_files):
        """
        Process only the drums.wav file from the separated files.
        
        :param separated_files: List of separated audio file paths.
        :return: List of processed file paths, where only drums.wav is processed.
        """
        processed_files = []
        for audio_file in separated_files:
            if "drums.wav" in audio_file:
                output_file = audio_file.replace(".wav", "_processed.wav")
                processed_files.append(self.process_file(audio_file, output_file))
            else:
                processed_files.append(audio_file)
        return processed_files
    
    # def process(self, input_file):
    #     """
    #     Process the input audio file and save the processed output to a file.
    #     """
    #     drum_f = ...
    #     piano_f = ...

    #     self.process_drum_files(drum_f)

    
    def get_out_fs(self):
        return sorted(glob.glob(osp.join(self.curr_save_dir, "*.wav")))
