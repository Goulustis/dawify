import os
import warnings
from typing import List, Optional, Set, Dict, Tuple, Counter
from dataclasses import dataclass, field
from mido import MidiFile, MidiTrack, Message, MetaMessage, second2tick, tick2second

MINIMUM_OFFSET_SEC = 0.01
MINIMUM_OFFSET_TIME = 0.01  # this is used to avoid zero-length notes

@dataclass
class Event:
    type: str
    value: int

@dataclass
class Note:
    is_drum: bool
    program: int  # MIDI program number (0-127)
    onset: float  # onset time in seconds
    offset: float  # offset time in seconds
    pitch: int  # MIDI note number (0-127)
    velocity: int  # (0-1) if ignore_velocity is True, otherwise (0-127)

@dataclass
class NoteEvent:
    is_drum: bool
    program: int  # [0, 127], 128 for drum but ignored in tokenizer
    time: Optional[float]  # absolute time in seconds (None for tie notes if needed)
    velocity: int  # currently 1 for onset, 0 for offset, drum has no offset
    pitch: int  # MIDI pitch
    activity: Optional[Set[int]] = field(default_factory=set)

    def equals_except(self, note_event, *excluded_attrs) -> bool:
        """ Check if two NoteEvent instances are equal EXCEPT for the 
        specified attributes. """
        if not isinstance(note_event, NoteEvent):
            return False

        for attr, value in self.__dict__.items():
            if attr not in excluded_attrs and value != note_event.__dict__.get(attr):
                return False
        return True

    def equals_only(self, note_event, *included_attrs) -> bool:
        """ Check if two NoteEvent instances are equal for the 
        specified attributes. """
        if not isinstance(note_event, NoteEvent):
            return False

        for attr in included_attrs:
            if self.__dict__.get(attr) != note_event.__dict__.get(attr):
                return False
        return True


def trim_overlapping_notes(notes: List[Note], sort: bool = True) -> List[Note]:
    """ Trim overlapping notes and dropping zero-length notes.
        https://github.com/magenta/mt3/blob/3deffa260ba7de3cf03cda1ea513a4d7ba7144ca/mt3/note_sequences.py#L52

        Trimming was only applied to train set, not test set in MT3.
    """
    if len(notes) <= 1:
        return notes

    trimmed_notes = []
    channels = set((note.pitch, note.program, note.is_drum) for note in notes)

    for pitch, program, is_drum in channels:
        channel_notes = [
            note for note in notes if note.pitch == pitch and note.program == program and note.is_drum == is_drum
        ]
        sorted_notes = sorted(channel_notes, key=lambda note: note.onset)

        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].offset > sorted_notes[i].onset:
                sorted_notes[i - 1].offset = sorted_notes[i].onset

        # Filter out zero-length notes
        valid_notes = [note for note in sorted_notes if note.onset < note.offset]

        trimmed_notes.extend(valid_notes)

    if sort:
        trimmed_notes.sort(key=lambda note: (note.onset, note.is_drum, note.program, note.velocity, note.pitch))
    return trimmed_notes


def validate_notes(notes: Tuple[List[Note]], minimum_offset: Optional[bool] = 0.01, fix: bool = True) -> List[Note]:
    """ validate and fix unrealistic notes """
    if len(notes) > 0:
        for note in list(notes):
            if note.onset == None:
                if fix:
                    notes.remove(note)
                continue
            elif note.offset == None:
                if fix:
                    note.offset = note.onset + MINIMUM_OFFSET_TIME
            elif note.onset > note.offset:
                warnings.warn(f'📙 Note at {note} has onset > offset.')
                if fix:
                    note.offset = max(note.offset, note.onset + MINIMUM_OFFSET_TIME)
                    print(f'✅\033[92m Fixed! Setting offset to onset + {MINIMUM_OFFSET_TIME}.\033[0m')
            elif note.is_drum is False and note.offset - note.onset < 0.01:
                # fix 13 Oct: too short notes issue for the dataset with non-MIDI annotations
                # warnings.warn(f'📙 Note at {note} has offset - onset < 0.01.')
                if fix:
                    note.offset = note.onset + MINIMUM_OFFSET_TIME
                    # print(f'✅\033[92m Fixed! Setting offset to onset + {MINIMUM_OFFSET_TIME}.\033[0m')

    return notes

def midi2note_event(input_file: os.PathLike,
                    assume_tempo: int = 500000,
                    default_ticks_per_beat: int = 480) -> List[NoteEvent]:
    """
    Converts a MIDI file to a list of NoteEvent objects.
    
    Because MIDI can have multiple tempo changes, for simplicity,
    this function assumes a single (or the first) tempo found, or
    `assume_tempo` if none is found. Ticks-per-beat is taken from 
    the MIDI file, or default if none is found.
    
    :param input_file: Path to the MIDI file to parse.
    :param assume_tempo: Fallback tempo in microseconds per quarter note.
    :param default_ticks_per_beat: Fallback ticks/PPQ if not in the file.
    :return: A list of NoteEvent objects.
    """
    midi = MidiFile(input_file)
    if midi.ticks_per_beat is not None:
        ticks_per_beat = midi.ticks_per_beat
    else:
        ticks_per_beat = default_ticks_per_beat

    # Look for the first tempo in the file, else assume_tempo
    tempo = None
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo is not None:
            break

    if tempo is None:
        tempo = assume_tempo

    note_events: List[NoteEvent] = []

    # We will track the current program *per channel* 
    # (since real MIDI can have multiple program changes).
    # Channel 9 is always drums, mapped to program=128 in our scheme.
    current_program_per_channel = dict((ch, 0) for ch in range(16))
    current_program_per_channel[9] = 128  # Drum channel → 128 internally

    # For absolute time accumulation
    # Each track has its own timeline that we’ll merge.
    # We'll store (absolute_seconds, channel, note_on|off, pitch, program).
    # Then sort by time in the end.
    event_buffer = []

    for track in midi.tracks:
        absolute_tick = 0
        # We can have a fallback program in this track if there's a "program_change" message.
        # But realistically, multiple program changes can appear. We'll pick the "latest" as we go.
        for msg in track:
            # Update absolute ticks
            absolute_tick += msg.time

            if msg.type == 'set_tempo':
                # If your MIDI has multiple tempo changes, you could store them in a list
                # or handle them properly. Here, we just override and keep going.
                tempo = msg.tempo

            elif msg.type == 'program_change':
                # Save the new program for that channel
                current_program_per_channel[msg.channel] = msg.program

            elif msg.type == 'note_on':
                # Note that a 'note_on' with velocity=0 is effectively a note_off
                on_velocity = msg.velocity
                is_on = (on_velocity > 0)

                # Convert ticks → seconds
                abs_seconds = tick2second(absolute_tick, ticks_per_beat, tempo)

                if msg.channel == 9:
                    # Drum channel
                    is_drum = True
                    program_ = 128
                else:
                    # Non-drum channel
                    is_drum = False
                    program_ = current_program_per_channel[msg.channel]

                if is_on:
                    # velocity=1 in our scheme
                    event_buffer.append((abs_seconds, is_drum, program_, 1, msg.note))
                else:
                    # velocity=0 in our scheme => offset
                    event_buffer.append((abs_seconds, is_drum, program_, 0, msg.note))

            elif msg.type == 'note_off':
                # Convert ticks → seconds
                abs_seconds = tick2second(absolute_tick, ticks_per_beat, tempo)

                if msg.channel == 9:
                    # Drum channel
                    is_drum = True
                    program_ = 128
                else:
                    # Non-drum channel
                    is_drum = False
                    program_ = current_program_per_channel[msg.channel]

                # velocity=0 in our scheme => offset
                event_buffer.append((abs_seconds, is_drum, program_, 0, msg.note))

            # If there are other messages, we ignore them in this simple approach.

    # Now we have a big buffer of events from all tracks.
    # We'll sort them by time primarily.
    event_buffer.sort(key=lambda x: x[0])

    # Turn into actual NoteEvent objects
    for abs_seconds, is_drum, program_, velocity, pitch in event_buffer:
        note_events.append(NoteEvent(
            is_drum=is_drum,
            program=program_,
            time=abs_seconds,
            velocity=velocity,
            pitch=pitch,
            activity=set()
        ))

    return note_events


def note_event2midi(note_events: List[NoteEvent],
                    output_file: Optional[os.PathLike] = None,
                    velocity: int = 100,
                    ticks_per_beat: int = 480,
                    tempo: int = 500000,
                    singing_program_mapping: int = 65,
                    singing_chorus_program_mapping: int = 53,
                    output_inverse_vocab: Optional[Dict] = None) -> None:
    """
    Converts a list of NoteEvent instances to a MIDI file.

    Example usage:
        note_event2midi(note_events, 'output.mid')
    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat, type=1)

    # Collect all programs from the note events
    programs = set()
    for ne in note_events:
        # 128 represents drum
        if ne.is_drum or ne.program == 128:
            programs.add(128)
        else:
            programs.add(ne.program)
    programs = sorted(programs)

    # Map each program to a channel
    program_to_channel = {}
    available_channels = list(range(0, 9)) + list(range(10, 16))
    for prg in programs:
        if prg == 128:
            program_to_channel[prg] = 9  # Drum channel
        else:
            try:
                program_to_channel[prg] = available_channels.pop(0)
            except IndexError:
                warnings.warn(f"No available channels for program {prg}, default to channel 15")
                program_to_channel[prg] = 15

    # For drum notes, we artificially create "offset" events
    # so that the MIDI plays a short note for each drum hit
    drum_offset_events = []
    for ne in note_events:
        if ne.is_drum and ne.velocity == 1:
            # Add a short offset 0.01s later
            drum_offset_events.append(
                NoteEvent(
                    is_drum=True,
                    program=128,
                    time=(ne.time + 0.01),
                    pitch=ne.pitch,
                    velocity=0,
                )
            )
    note_events_extended = note_events + drum_offset_events
    # Sort all events by time
    note_events_extended.sort(key=lambda ne: (ne.time, ne.is_drum, ne.program, ne.velocity, ne.pitch))

    # For each program, we create a separate track
    for program in programs:
        track = MidiTrack()
        midi.tracks.append(track)

        # Track name
        if program == 128:
            program_name = "Drums"
        elif output_inverse_vocab is not None:
            program_name = output_inverse_vocab.get(program, (program, f"Prg. {str(program)}"))[1]
        else:
            program_name = f"Prg. {program}"
        track.append(MetaMessage("track_name", name=program_name, time=0))

        # Determine the MIDI channel
        channel = program_to_channel[program]

        # Program change message for this track
        if program == 128:
            # Drums: channel 9. The "program" is ignored by GM on channel 9 anyway
            track.append(Message('program_change', program=0, time=0, channel=channel))
        elif program == 100:
            # special singing program -> Alto Sax
            track.append(Message('program_change', program=singing_program_mapping, time=0, channel=channel))
        elif program == 101:
            # special singing chorus program -> Voice Oohs
            track.append(Message('program_change', program=singing_chorus_program_mapping, time=0, channel=channel))
        else:
            track.append(Message('program_change', program=program, time=0, channel=channel))

        # We'll add note on/off messages for those note events that match this program
        current_tick = 0
        for ne in note_events_extended:
            if ne.program == program:
                # Convert absolute time (seconds) -> ticks
                absolute_tick = round(second2tick(ne.time, ticks_per_beat, tempo))
                delta_tick = max(0, absolute_tick - current_tick)
                current_tick += delta_tick

                if ne.velocity > 0:
                    # note_on
                    msg_type = "note_on"
                    msg_velocity = velocity
                else:
                    # note_off
                    msg_type = "note_off"
                    msg_velocity = 0

                track.append(Message(
                    msg_type,
                    note=ne.pitch,
                    velocity=msg_velocity,
                    time=delta_tick,
                    channel=channel
                ))

    # Write out the file if requested
    if output_file is not None:
        midi.save(output_file)


def test_midi_conversion(original_midi_file: os.PathLike,
                         temp_midi_file: os.PathLike = "temp_output.mid") -> bool:
    """
    Reads a MIDI file -> converts to NoteEvent -> writes those NoteEvent 
    to a MIDI file -> reads that new MIDI file back to NoteEvent -> 
    compares that both NoteEvent lists are (reasonably) the same.
    
    :param original_midi_file: Path to the original MIDI file.
    :param temp_midi_file: Path where the intermediate MIDI will be saved.
    :return: True if the round-trip conversion is (reasonably) identical, False otherwise.
    """
    # Step 1: MIDI -> NoteEvent
    note_events_1 = midi2note_event(original_midi_file)
    
    # Step 2: NoteEvent -> MIDI
    note_event2midi(note_events_1, temp_midi_file)
    
    # Step 3: MIDI -> NoteEvent again
    note_events_2 = midi2note_event(temp_midi_file)
    
    if len(note_events_1) != len(note_events_2):
        print("Different number of events between original and round-trip note_events.")
        return False
    
    # Sort them by (time, pitch, velocity, program, is_drum) for stable comparison
    def sort_key(e: NoteEvent):
        return (round(e.time, 3), e.pitch, e.velocity, e.program, e.is_drum)
    
    # We might allow some minor floating time differences. Let's round or 
    # check if they're close enough.
    sorted_1 = sorted(note_events_1, key=sort_key)
    sorted_2 = sorted(note_events_2, key=sort_key)
    
    for e1, e2 in zip(sorted_1, sorted_2):
        # Check equality on the main fields. Times might differ by small floating error.
        if abs(e1.time - e2.time) > 1e-3:  # allow small difference
            print(f"Time mismatch: {e1.time} vs {e2.time}")
            return False
        if e1.pitch != e2.pitch:
            print(f"Pitch mismatch: {e1.pitch} vs {e2.pitch}")
            return False
        if e1.velocity != e2.velocity:
            print(f"Velocity mismatch: {e1.velocity} vs {e2.velocity}")
            return False
        if e1.is_drum != e2.is_drum:
            print(f"Drum mismatch: {e1.is_drum} vs {e2.is_drum}")
            return False
        if e1.program != e2.program:
            print(f"Program mismatch: {e1.program} vs {e2.program}")
            return False

    print("Round-trip conversion is consistent!")
    return True


def note2note_event(notes: List[Note], sort: bool = True, return_activity: bool = True) -> List[NoteEvent]:
    """
    note2note_event:
    Converts a list of Note instances to a list of NoteEvent instances.

    Args:
    - notes (List[Note]): A list of Note instances.
    - sort (bool): Sort the NoteEvent instances by increasing order of onsets,
      and at the same timing, by increasing order of program and pitch.
      Default is True. If return_activity is set to True, NoteEvent instances
      are sorted regardless of this argument.
    - return_activity (bool): If True, return a list of NoteActivity instances

    Returns:
    - note_events (List[NoteEvent]): A list of NoteEvent instances.

    """
    note_events = []
    for note in notes:
        # for each note, add onset and offset events
        note_events.append(NoteEvent(note.is_drum, note.program, note.onset, note.velocity, note.pitch))
        if note.is_drum == 0:  # (drum has no offset!)
            note_events.append(NoteEvent(note.is_drum, note.program, note.offset, 0, note.pitch))

    if sort or return_activity:
        note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))

    if return_activity:
        # activity stores the indices of previous notes that are still active
        activity = set()  # mutable class
        for i, ne in enumerate(note_events):
            # set a copy of the activity set ti the current note event
            ne.activity = activity.copy()

            if ne.is_drum:
                continue  # drum's offset and activity are not tracked
            elif ne.velocity == 1:
                activity.add(i)
            elif ne.velocity == 0:
                # search for the index of matching onset event
                matched_onset_event_index = None
                for j in activity:
                    if note_events[j].equals_only(ne, 'is_drum', 'program', 'pitch'):
                        matched_onset_event_index = j
                        break
                if matched_onset_event_index is not None:
                    activity.remove(matched_onset_event_index)
                else:
                    raise ValueError(f'📕 note2note_event: no matching onset event for {ne}')
            else:
                raise ValueError(f'📕 Invalid velocity: {ne.velocity} expected 0 or 1')
        if len(activity) > 0:
            # if there are still active notes at the end of the sequence
            warnings.warn(f'📙 note2note_event: {len(activity)} notes are still \
                          active at the end of the sequence. Please validate \
                          the input Note instances. ')
    return note_events


def note_event2event(note_events: List[NoteEvent],
                     tie_note_events: Optional[List[NoteEvent]] = None,
                     start_time: float = 0.,
                     tps: int = 100,
                     sort: bool = True) -> List[Event]:
    """ note_event2event:
    Converts a list of NoteEvent instances to a list of Event instances.
    - NoteEvent instances have absolute time within a file, while Event instances
        have 'shift' events of absolute time within a segment.
    - Tie NoteEvent instances are prepended to output list of Event instances,
        and closed by a 'tie' event.
    - If start_time is not provided, start_time=0 in seconds by default. 
    - If there is non-tie note_event instances before the start_time, raises an error.

    Args:
    - note_events (list[NoteEvent]): A list of NoteEvent instances.
    - tie_note_events (Optional[list[NoteEvent]]): A list of tie NoteEvent instances.
        See slice_note_events_and_ties() for more details. Default is None.
    - start_time (float): Start time in seconds. Default is 0. Any non-tie NoteEvent 
        instances should have time >= start_time. 
    - tps (Optional[int]): Ticks per second. Default is 100.
    - sort (bool): If True, sort the Event instances by increasing order of
        onsets, and at the same timing, by increasing order of program and pitch.
        Default is False.

    Returns:
    - events (list[Event]): A list of Event instances.
    """
    if sort:
        if tie_note_events != None:
            tie_note_events.sort(key=lambda n_ev: (n_ev.program, n_ev.pitch))
        note_events.sort(
            key=lambda n_ev: (round(n_ev.time * tps), n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))

    # Initialize event list and state variables
    events = []
    start_tick = round(start_time * tps)
    tick_state = start_tick

    program_state = None

    # Prepend tie events
    if tie_note_events:
        for tne in tie_note_events:
            if tne.program != program_state:
                events.append(Event(type='program', value=tne.program))
                program_state = tne.program
            events.append(Event(type='pitch', value=tne.pitch))

    # Any tie events (can be empty) are closed by a 'tie' event
    events.append(Event(type='tie', value=0))

    # Translate NoteEvent to Event in the list
    velocity_state = None  # reset state variables
    for ne in note_events:
        if ne.is_drum and ne.velocity == 0:  # <-- bug fix
            continue  # drum's offset should be ignored, and should not cause shift

        # Process time shift and update tick_state
        ne_tick = round(ne.time * tps)
        if ne_tick > tick_state:
            # shift_ticks = ne_tick - tick_state
            shift_ticks = ne_tick - start_tick
            events.append(Event(type='shift', value=shift_ticks))
            tick_state = ne_tick
        elif ne_tick == tick_state:
            pass
        else:
            raise ValueError(
                f'NoteEvent tick_state {ne_tick} of time {ne.time} is smaller than tick_state {tick_state}.')

        # Process program change and update program_state
        if ne.is_drum and ne.velocity == 1:
            # drum events have no program and offset but velocity 1
            if velocity_state != 1 or velocity_state == None:
                events.append(Event(type='velocity', value=1))
                velocity_state = 1
            events.append(Event(type='drum', value=ne.pitch))
        else:
            if ne.program != program_state or program_state == None:
                events.append(Event(type='program', value=ne.program))
                program_state = ne.program

            if ne.velocity != velocity_state or velocity_state == None:
                events.append(Event(type='velocity', value=ne.velocity))
                velocity_state = ne.velocity

            events.append(Event(type='pitch', value=ne.pitch))

    return events

def note2note_event(notes: List[Note], sort: bool = True, return_activity: bool = True) -> List[NoteEvent]:
    """
    note2note_event:
    Converts a list of Note instances to a list of NoteEvent instances.

    Args:
    - notes (List[Note]): A list of Note instances.
    - sort (bool): Sort the NoteEvent instances by increasing order of onsets,
      and at the same timing, by increasing order of program and pitch.
      Default is True. If return_activity is set to True, NoteEvent instances
      are sorted regardless of this argument.
    - return_activity (bool): If True, return a list of NoteActivity instances

    Returns:
    - note_events (List[NoteEvent]): A list of NoteEvent instances.

    """
    note_events = []
    for note in notes:
        # for each note, add onset and offset events
        note_events.append(NoteEvent(note.is_drum, note.program, note.onset, note.velocity, note.pitch))
        if note.is_drum == 0:  # (drum has no offset!)
            note_events.append(NoteEvent(note.is_drum, note.program, note.offset, 0, note.pitch))

    if sort or return_activity:
        note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))

    if return_activity:
        # activity stores the indices of previous notes that are still active
        activity = set()  # mutable class
        for i, ne in enumerate(note_events):
            # set a copy of the activity set ti the current note event
            ne.activity = activity.copy()

            if ne.is_drum:
                continue  # drum's offset and activity are not tracked
            elif ne.velocity == 1:
                activity.add(i)
            elif ne.velocity == 0:
                # search for the index of matching onset event
                matched_onset_event_index = None
                for j in activity:
                    if note_events[j].equals_only(ne, 'is_drum', 'program', 'pitch'):
                        matched_onset_event_index = j
                        break
                if matched_onset_event_index is not None:
                    activity.remove(matched_onset_event_index)
                else:
                    raise ValueError(f'📕 note2note_event: no matching onset event for {ne}')
            else:
                raise ValueError(f'📕 Invalid velocity: {ne.velocity} expected 0 or 1')
        if len(activity) > 0:
            # if there are still active notes at the end of the sequence
            warnings.warn(f'📙 note2note_event: {len(activity)} notes are still \
                          active at the end of the sequence. Please validate \
                          the input Note instances. ')
    return note_events


def slice_note_events_and_ties(note_events: List[NoteEvent],
                               start_time: float,
                               end_time: float,
                               tidyup: bool = False) -> Tuple[List[NoteEvent], List[NoteEvent], int]:
    """
    Extracts a specific subsequence of note events and tie note events for the
    first note event in the subsequence.
    
    Args:
    - note_events (List[NoteEvent]): List of NoteEvent instances.
    - start_time (float): The start time of the subsequence in seconds.
    - end_time (float): The end time of the subsequence in seconds.
    - tidyup (Optional[bool]): If True, sort the resulting lists of NoteEvents,
        and remove the activity attribute of sliced_note_event, and remove the
        time and activity attributes of tie_note_events. Default is False.
        Avoid using tidyup=True without deepcopying the original note_events.

    Note:
    - The activity attribute of returned sliced_note_events, and the time and
      activity attributes of tie_note_events are not valid after slicing. 
      Thus, they should be ignored in the downstream processing. 

    Returns:
    - sliced_note_events (List[NoteEvent]): List of NoteEvent instances in the
                                            specified range.
    - tie_note_events (List[NoteEvent]): List of NoteEvent instances that are
                                          active (tie) at start_time.
    - start_time (float): Just bypass the start time from the input argument.
    """
    if start_time > end_time:
        raise ValueError(f'📕 slice_note_events: start_time {start_time} \
                          is greater than end_time {end_time}')
    elif len(note_events) == 0:
        warnings.warn('📙 slice_note_events: empty note_events as input')
        return [], [], start_time

    # Get start_index and end_index
    start_index, end_index = None, None
    found_start = False
    for i, ne in enumerate(note_events):
        if not found_start and ne.time >= start_time and ne.time < end_time:
            start_index = i
            found_start = True

        if ne.time >= end_time:
            end_index = i
            break

    # Get tie_note_events
    if start_index == None:
        if end_index == 0:
            tie_note_events = []
        elif end_index == None:
            tie_note_events = []
        else:
            tie_note_events = [note_events[i] for i in note_events[end_index].activity]
    else:
        tie_note_events = [note_events[i] for i in note_events[start_index].activity]
    """ modifying note events here is dangerous, due to mutability of original note_events!! """
    if tidyup:
        for tne in tie_note_events:
            tne.time = None
            tne.activity = None

    tie_note_events.sort(key=lambda n_ev: (n_ev.program, n_ev.pitch))

    # Get sliced note_events
    if start_index is None:
        sliced_note_events = []
    else:
        sliced_note_events = note_events[start_index:end_index]

    if tidyup:
        for sne in sliced_note_events:
            sne.activity = None

    sliced_note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))
    return sliced_note_events, tie_note_events, start_time


def note_event2note(
    note_events: List[NoteEvent],
    tie_note_events: Optional[List[NoteEvent]] = None,
    sort: bool = True,
    fix_offset: bool = True,
    trim_overlap: bool = True,
) -> Tuple[List[Note], Counter[str]]:
    """Convert note events to notes.

    Returns:
        List[Note]: A list of merged note events.
        Counter[str]: A dictionary of error counters.
    """

    notes = []
    active_note_events = {}

    error_counter = {}  # Add a dictionary to count the errors by their types

    if tie_note_events is not None:
        for ne in tie_note_events:
            active_note_events[(ne.pitch, ne.program)] = ne

    if sort:
        note_events.sort(key=lambda ne: (ne.time, ne.is_drum, ne.pitch, ne.velocity, ne.program))

    for ne in note_events:
        try:
            if ne.time == None:
                continue
            elif ne.is_drum:
                if ne.velocity == 1:
                    notes.append(
                        Note(is_drum=True,
                             program=128,
                             onset=ne.time,
                             offset=ne.time + MINIMUM_OFFSET_SEC,
                             pitch=ne.pitch,
                             velocity=1))
                else:
                    continue
            elif ne.velocity == 1:
                active_ne = active_note_events.get((ne.pitch, ne.program))
                if active_ne is not None:
                    active_note_events.pop((ne.pitch, ne.program))
                    notes.append(
                        Note(False, active_ne.program, active_ne.time, ne.time, active_ne.pitch, active_ne.velocity))
                active_note_events[(ne.pitch, ne.program)] = ne

            elif ne.velocity == 0:
                active_ne = active_note_events.pop((ne.pitch, ne.program), None)
                if active_ne is not None:
                    notes.append(
                        Note(False, active_ne.program, active_ne.time, ne.time, active_ne.pitch, active_ne.velocity))
                else:
                    raise ValueError('Err/onset not found')
        except ValueError as ve:
            error_type = str(ve)
            error_counter[error_type] = error_counter.get(error_type, 0.) + 1

    for ne in active_note_events.values():
        try:
            if ne.velocity == 1:
                if ne.program == None or ne.pitch == None:
                    raise ValueError('Err/active ne incomplete')
                elif ne.time == None:
                    continue
                else:
                    notes.append(
                        Note(is_drum=False,
                             program=ne.program,
                             onset=ne.time,
                             offset=ne.time + MINIMUM_OFFSET_SEC,
                             pitch=ne.pitch,
                             velocity=1))
        except ValueError as ve:
            error_type = str(ve)
            error_counter[error_type] = error_counter.get(error_type, 0.) + 1

    if fix_offset:
        for n in list(notes):
            try:
                if n.offset - n.onset > 10:
                    n.offset = n.onset + MINIMUM_OFFSET_SEC
                    raise ValueError('Err/long note > 10s')
            except ValueError as ve:
                error_type = str(ve)
                error_counter[error_type] = error_counter.get(error_type, 0.) + 1

    if sort:
        notes.sort(key=lambda note: (note.onset, note.is_drum, note.program, note.velocity, note.pitch))

    if fix_offset:
        notes = validate_notes(notes, fix=True)

    if trim_overlap:
        notes = trim_overlapping_notes(notes, sort=True)

    return notes, error_counter

def cu_midi2note(inp_f):
    note_evs = midi2note_event(inp_f)
    notes, err_cnt = note_event2note(note_evs)

    return notes


if __name__ == "__main__":
    # Test the MIDI conversion
    inp_f = "/home/boss/projects/dawify/assets/sample_level1/sample_level1_allMIDI.mid"
    ls_notes = midi2note_event(inp_f)
    note_event2midi(ls_notes, "output.mid")
    test_midi_conversion(inp_f, "output.mid")