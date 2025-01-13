import librosa
import numpy as np


class AudioRhythmAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def estimate_tempo(audio_file: str) -> (float, np.ndarray):
        # Load the audio file
        y, sr = librosa.load(audio_file)

        # Perform beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        return tempo, beat_times
