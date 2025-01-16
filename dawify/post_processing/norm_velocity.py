import mido

from loguru import logger

from dawify.post_processing.base import MidiPostProcessorBase


class VelocityNormalizer(MidiPostProcessorBase):
    """Processor for normalizing note velocities"""
    def _get_average_velocity(self, track: mido.MidiTrack) -> float:
        velocities = []
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                velocities.append(msg.velocity)
        avg = sum(velocities) / len(velocities) if velocities else 80
        logger.debug(f'Average velocity: {avg}')
        return avg

    def process(self, min_velocity: int = 40) -> None:
        """Normalize velocities in all tracks."""
        for i, track in enumerate(self.midi.tracks):
            logger.info(f'Normalizing velocities for track {i}')
            avg_velocity = self._get_average_velocity(track)

            for msg in track:
                if msg.type != 'note_on':
                    continue

                if msg.velocity <= 0 or msg.velocity >= min_velocity:
                    continue

                old_velocity = msg.velocity
                msg.velocity = int(avg_velocity)
                logger.debug(f'Adjusted velocity from {old_velocity} to {msg.velocity}')
