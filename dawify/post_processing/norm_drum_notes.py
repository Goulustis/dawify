from loguru import logger

from dawify.post_processing.base import MidiPostProcessorBase
from dawify.post_processing.utils import is_drum_track, DrumNotes


class DrumNoteNormalizer(MidiPostProcessorBase):
    """Processor for normalizing different types of drum notes to a standard set"""
    def _normalize_note(self, note: int, note_map: dict[int, str], target_note: int) -> int:
        """Normalize a note to a target note if it's in the note map."""
        return target_note if note in note_map else note

    def process(self):
        """Normalize different types of drum notes to standard ones."""
        for i, track in enumerate(self.midi.tracks):
            logger.info(f'Normalizing drum notes for track {i}')
            if not is_drum_track(track):
                logger.debug(f'Skipping non-drum track {i}')
                continue

            for j, msg in enumerate(track):
                if msg.type == 'note_on' or msg.type == 'note_off':
                    # Normalize hi-hats to closed hi-hat
                    msg.note = self._normalize_note(msg.note, DrumNotes.HIHAT, 42)
                    # Normalize rides to ride cymbal 1
                    msg.note = self._normalize_note(msg.note, DrumNotes.RIDE, 51)
                    # Normalize snares to acoustic snare
                    msg.note = self._normalize_note(msg.note, DrumNotes.SNARE, 38)
                    # Normalize toms to low tom
                    msg.note = self._normalize_note(msg.note, DrumNotes.TOM, 45)
                else:
                    logger.debug(f'Skipping non-note message in track {i}, bar {j}')

        logger.info('Drum notes normalized successfully.')
