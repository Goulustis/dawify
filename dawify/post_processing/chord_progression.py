import crema
import pandas as pd


class ChordProgression:

    MODEL = crema.models.chord.ChordModel()

    def __init__(self):
        pass

    @staticmethod
    def extract_chords(audio_path: str) -> pd.DataFrame:
        """
        Chord recognition using CREMA.
        """
        chord_est = ChordProgression.MODEL.predict(filename=audio_path)
        return chord_est.to_dataframe()
