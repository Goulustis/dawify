import mido


class DrumNotes:
    """Static class containing MIDI note mappings for drums"""
    HIHAT = {
        42: 'Closed Hi-Hat',
        44: 'Pedal Hi-Hat',
        46: 'Open Hi-Hat',
    }

    RIDE = {
        51: 'Ride Cymbal 1',
        53: 'Ride Bell',
        59: 'Ride Cymbal 2',
    }

    SNARE = {
        38: 'Acoustic Snare',
        40: 'Electric Snare',
    }

    TOM = {
        41: 'Low Floor Tom',
        43: 'High Floor Tom',
        45: 'Low Tom',
        47: 'Low-Mid Tom',
        48: 'Hi-Mid Tom',
        50: 'High Tom',
    }


def is_drum_track(track: mido.MidiTrack) -> bool:
    """Check if a track is a drum track (channel 9)."""
    for msg in track:
        if hasattr(msg, 'channel') and msg.channel == 9:
            return True
    return False
