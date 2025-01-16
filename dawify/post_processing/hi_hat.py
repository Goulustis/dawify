from collections import Counter

import mido
from loguru import logger

from dawify.post_processing.base import MidiPostProcessorBase
from dawify.post_processing.utils import is_drum_track, DrumNotes


def _get_hihat_pattern(
    track: mido.MidiTrack,
    bar_start: int,
    bar_end: int,
    bar_index: int,
) -> list[tuple[int, int]]:
    """Get hi-hat and ride pattern for a specific bar."""
    current_time = 0
    bar_hits = []

    # First pass: collect all hits within the bar's time range
    for msg in track:
        current_time += msg.time
        if (msg.type == 'note_on' and
            (msg.note in DrumNotes.HIHAT or msg.note in DrumNotes.RIDE) and
            bar_start <= current_time < bar_end
        ):
            # Calculate position within bar as a fraction (0 to 1)
            relative_position = (current_time - bar_start) / (bar_end - bar_start)
            # Quantize to nearest 16th note (16 divisions per bar)
            quantized_position = round(relative_position * 16) / 16
            # Convert back to ticks
            quantized_time = int(quantized_position * (bar_end - bar_start))
            bar_hits.append((quantized_time, msg.note))

    # Sort hits by time to ensure correct ordering
    bar_hits.sort(key=lambda x: x[0])

    # Remove duplicates that might have been created by quantization
    unique_hits = []
    prev_time = None
    for time, note in bar_hits:
        if time != prev_time:
            unique_hits.append((time, note))
            prev_time = time

    logger.debug(f'Bar {bar_index}: Hi-hat/ride pattern [{bar_start}-{bar_end}]: {unique_hits}')
    return unique_hits


def _has_toms(track: mido.MidiTrack, bar_start: int, bar_end: int, bar_index: int) -> bool:
    """Check if a bar contains any tom hits."""
    current_time = 0
    for msg in track:
        current_time += msg.time
        if (msg.type == 'note_on' and
            msg.note in DrumNotes.TOM and
            bar_start <= current_time < bar_end
        ):
            logger.debug(
                f'Bar {bar_index}: Found tom {msg.note} ({DrumNotes.TOM.get(msg.note)}) at time {current_time}',
            )
            return True
    return False


def _get_normalized_intervals(pattern: list[tuple[int, int]]) -> list[int]:
    if len(pattern) < 2:
        return []

    # Calculate raw intervals
    intervals = []
    for i in range(1, len(pattern)):
        interval = pattern[i][0] - pattern[i-1][0]
        intervals.append(interval)

    # Find the most common interval (base interval)
    interval_counts = Counter(intervals)
    base_interval = interval_counts.most_common(1)[0][0]

    # Normalize intervals to multiples of base interval
    normalized = []
    for interval in intervals:
        ratio = round(interval / base_interval)
        normalized.append(ratio)

    logger.debug(f'Raw intervals: {intervals}, Base interval: {base_interval}, Normalized: {normalized}')
    return normalized


def _patterns_match(
        pattern1: list[tuple[int, int]],
        pattern2: list[tuple[int, int]],
        prev_bar_idx: int,
        next_bar_idx: int,
) -> bool:
    """Check if two hi-hat/ride patterns match by comparing normalized intervals."""
    logger.debug(f'Comparing patterns between bars {prev_bar_idx} and {next_bar_idx}:')
    logger.debug(
        f'Bar {prev_bar_idx}: {[(t, DrumNotes.HIHAT.get(n, DrumNotes.RIDE.get(n, n))) for t, n in pattern1]}',
    )
    logger.debug(
        f'Bar {next_bar_idx}: {[(t, DrumNotes.HIHAT.get(n, DrumNotes.RIDE.get(n, n))) for t, n in pattern2]}',
    )

    if len(pattern1) != len(pattern2):
        logger.debug(
            f'Patterns have different lengths: '
            f'bar {prev_bar_idx}={len(pattern1)}, '
            f'bar {next_bar_idx}={len(pattern2)}',
        )
        return False

    norm1 = _get_normalized_intervals(pattern1)
    norm2 = _get_normalized_intervals(pattern2)

    if norm1 != norm2:
        logger.debug(f'Normalized pattern mismatch: {norm1} != {norm2}')
        return False

    logger.debug(f'Patterns match between bars {prev_bar_idx} and {next_bar_idx}')
    return True


class HiHatProcessor(MidiPostProcessorBase):
    """Processor for hi-hat pattern matching and updating"""

    def process(self) -> None:
        """Process hi-hats and rides in drum tracks according to the specified rules."""
        for track_idx, track in enumerate(self.midi.tracks):
            if not is_drum_track(track):
                logger.info(f'Processing track {track_idx} (Drum track: False)')
                continue

            logger.info(f'Processing track {track_idx} (Drum track: True)')

            # Process 8-bar regions
            total_ticks = sum(msg.time for msg in track)
            region_size = self.ticks_per_bar * 8
            total_bars = total_ticks // self.ticks_per_bar + (1 if total_ticks % self.ticks_per_bar else 0)

            logger.info(f'Total ticks: {total_ticks}, Region size: {region_size}, Total bars: {total_bars}')

            for region_start in range(0, total_ticks, region_size):
                region_end = min(region_start + region_size, total_ticks)
                region_idx = region_start // region_size
                region_start_bar = region_start // self.ticks_per_bar
                region_end_bar = min(region_end // self.ticks_per_bar, total_bars)

                logger.info(f'Processing region {region_idx} (bars {region_start_bar}-{region_end_bar})')

                # Process bars 2-7 in the region
                for bar_idx in range(1, 7):
                    bar_start = region_start + (bar_idx * self.ticks_per_bar)
                    bar_end = bar_start + self.ticks_per_bar
                    absolute_bar_idx = region_start_bar + bar_idx

                    if bar_end > region_end:
                        logger.debug(f'Bar {absolute_bar_idx} end {bar_end} exceeds region end {region_end}, skipping')
                        break

                    logger.info(f'Processing bar {absolute_bar_idx} (region {region_idx}, position {bar_idx+1}/8)')

                    # Get patterns for current, previous, and next bars
                    prev_bar_start = bar_start - self.ticks_per_bar
                    next_bar_start = bar_end
                    next_bar_end = next_bar_start + self.ticks_per_bar

                    # Check for toms in previous and next bars
                    has_prev_toms = _has_toms(track, prev_bar_start, bar_start, absolute_bar_idx-1)
                    has_next_toms = _has_toms(track, next_bar_start, next_bar_end, absolute_bar_idx+1)

                    if has_prev_toms or has_next_toms:
                        logger.debug(
                            f'Found toms in adjacent bars '
                            f'({absolute_bar_idx-1} or {absolute_bar_idx+1}), '
                            f'skipping pattern matching',
                        )
                        continue

                    prev_pattern = _get_hihat_pattern(track, prev_bar_start, bar_start, absolute_bar_idx-1)
                    next_pattern = _get_hihat_pattern(track, next_bar_start, next_bar_end, absolute_bar_idx+1)

                    # If previous and next bars match, update current bar
                    if not (
                        prev_pattern
                        and next_pattern
                        and _patterns_match(
                            prev_pattern,
                            next_pattern,
                            absolute_bar_idx-1,
                            absolute_bar_idx+1,
                        )
                    ):
                        continue

                    logger.info(
                        f'Patterns match between bars {absolute_bar_idx-1} and {absolute_bar_idx+1}, '
                        f'updating bar {absolute_bar_idx}',
                    )

                    # Remove existing hi-hats and rides in current bar
                    messages_to_remove = []
                    messages_to_add = []
                    current_time = 0

                    for i, msg in enumerate(track):
                        current_time += msg.time
                        if (
                            bar_start <= current_time < bar_end
                            and msg.type in ['note_on', 'note_off']
                            and (msg.note in DrumNotes.HIHAT or msg.note in DrumNotes.RIDE)
                        ):
                            messages_to_remove.append(i)

                    # Add new hi-hat pattern
                    for time_offset, note in prev_pattern:
                        # Create note_on message
                        msg = mido.Message('note_on', note=note, velocity=100, time=time_offset, channel=9)
                        messages_to_add.append((bar_start + time_offset, msg))
                        # Create note_off message
                        msg_off = mido.Message('note_off', note=note, velocity=0, time=10, channel=9)
                        messages_to_add.append((bar_start + time_offset + 10, msg_off))

                    # Remove old messages in reverse order
                    for i in reversed(messages_to_remove):
                        del track[i]

                    # Add new messages
                    messages_to_add.sort(key=lambda x: x[0])
                    for _, msg in messages_to_add:
                        track.append(msg)


if __name__ == '__main__':
    input_path = '/users/ken/_input/sample_level1_pred_input/drums.mid'
    output_path = '/users/ken/_output/sample_level1_pred_output/drums.mid'

    processor = HiHatProcessor(input_path)
    processor.process()
    processor.save(output_path)
