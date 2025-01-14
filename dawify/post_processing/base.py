from pathlib import Path

import mido
from loguru import logger


class MidiPostProcessorBase:
    """Base class for MIDI post-processors"""
    def __init__(self, midi_file_path: str):
        self.midi_file_path = midi_file_path
        self.midi = mido.MidiFile(midi_file_path)
        self.ticks_per_bar = None
        self._calculate_ticks_per_bar()

    def _calculate_ticks_per_bar(self) -> None:
        """Calculate ticks per bar based on time signature and ticks per beat."""
        time_sig = None
        for track in self.midi.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    time_sig = msg
                    break
            if time_sig:
                break

        if time_sig:
            self.ticks_per_bar = self.midi.ticks_per_beat * time_sig.numerator
            logger.debug(f'Found time signature: {time_sig.numerator}/{time_sig.denominator}')
        else:
            self.ticks_per_bar = self.midi.ticks_per_beat * 4
            logger.debug('No time signature found, defaulting to 4/4')

        logger.debug(f'Ticks per bar: {self.ticks_per_bar}')

    def save(self, output_path: str) -> None:
        """Save the processed MIDI file"""
        logger.info(f'Saving to {output_path}')

        if path := Path(output_path).parent:
            path.mkdir(parents=True, exist_ok=True)

        self.midi.save(output_path)

        logger.info('Saved!')
