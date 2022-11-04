import numpy as np
from numba import jit
import pretty_midi
import scipy.interpolate as interp

TOKEN_SPECIAL: int = 0
TOKEN_NOTE: int = 1
TOKEN_VELOCITY: int = 2
TOKEN_TIME: int = 3

DEFAULT_VELOCITY: int = 77

TIE: int = 2
EOS: int = 1
PAD: int = 0


def extrapolate_beat_times(beat_times, n_extend=1):
    beat_times_function = interp.interp1d(
        np.arange(beat_times.size),
        beat_times,
        bounds_error=False,
        fill_value="extrapolate",
    )

    ext_beats = beat_times_function(
        np.linspace(0, beat_times.size + n_extend - 1, beat_times.size + n_extend)
    )

    return ext_beats


@jit(nopython=True, cache=True)
def fast_tokenize(idx, token_type, n_special, n_note, n_velocity):
    if token_type == TOKEN_TIME:
        return n_special + n_note + n_velocity + idx
    elif token_type == TOKEN_VELOCITY:
        return n_special + n_note + idx
    elif token_type == TOKEN_NOTE:
        return n_special + idx
    elif token_type == TOKEN_SPECIAL:
        return idx
    else:
        return -1


@jit(nopython=True, cache=True)
def fast_detokenize(idx, n_special, n_note, n_velocity, time_idx_offset):
    if idx >= n_special + n_note + n_velocity:
        return (TOKEN_TIME, (idx - (n_special + n_note + n_velocity)) + time_idx_offset)
    elif idx >= n_special + n_note:
        return TOKEN_VELOCITY, idx - (n_special + n_note)
    elif idx >= n_special:
        return TOKEN_NOTE, idx - n_special
    else:
        return TOKEN_SPECIAL, idx


class MidiTokenizer:
    def __init__(self, config) -> None:
        self.config = config

    def tokenize_note(self, idx, token_type):
        rt = fast_tokenize(
            idx,
            token_type,
            self.config.vocab_size.special,
            self.config.vocab_size.note,
            self.config.vocab_size.velocity,
        )
        if rt == -1:
            raise ValueError(f"type {type} is not a predefined token type.")
        else:
            return rt

    def notes_to_tokens(self, notes):
        """
        notes : (onset idx, offset idx, pitch, velocity)
        """
        max_time_idx = notes[:, :2].max()

        times = [[] for i in range((max_time_idx + 1))]
        for onset, offset, pitch, velocity in notes:
            times[onset].append([pitch, velocity])
            times[offset].append([pitch, 0])

        tokens = []
        current_velocity = 0
        for i, time in enumerate(times):
            if len(time) == 0:
                continue
            tokens.append(self.tokenize_note(i, TOKEN_TIME))
            for pitch, velocity in time:
                velocity = int(velocity > 0)
                if current_velocity != velocity:
                    current_velocity = velocity
                    tokens.append(self.tokenize_note(velocity, TOKEN_VELOCITY))
                tokens.append(self.tokenize_note(pitch, TOKEN_NOTE))

        return np.array(tokens, dtype=int)

    def detokenize(self, token, time_idx_offset):
        type, value = fast_detokenize(
            token,
            n_special=self.config.vocab_size.special,
            n_note=self.config.vocab_size.note,
            n_velocity=self.config.vocab_size.velocity,
            time_idx_offset=time_idx_offset,
        )
        if type != TOKEN_TIME:
            value = int(value)
        return [type, value]

    def to_string(self, tokens, time_idx_offset=0):
        nums = [
            self.detokenize(token, time_idx_offset=time_idx_offset) for token in tokens
        ]
        strings = []
        for i in range(len(nums)):
            type = nums[i][0]
            value = nums[i][1]

            if type == TOKEN_TIME:
                type = "time"
            elif type == TOKEN_SPECIAL:
                if value == EOS:
                    value = "EOS"
                elif value == PAD:
                    value = "PAD"
                elif value == TIE:
                    value = "TIE"
                else:
                    value = "Unknown Special"
            elif type == TOKEN_NOTE:
                type = "note"
            elif type == TOKEN_VELOCITY:
                type = "velocity"
            strings.append((type, value))
        return strings

    def split_notes(self, notes, beatsteps, time_from, time_to):
        """
        Assumptions
        - notes are sorted by onset time
        - beatsteps are sorted by time
        """
        start_idx = np.searchsorted(beatsteps, time_from)
        start_note = np.searchsorted(notes[:, 0], start_idx)

        end_idx = np.searchsorted(beatsteps, time_to)
        end_note = np.searchsorted(notes[:, 0], end_idx)
        splited_notes = notes[start_note:end_note]

        return splited_notes, (start_idx, end_idx, start_note, end_note)

    def notes_to_relative_tokens(
        self, notes, offset_idx, add_eos=False, add_composer=False, composer_value=None
    ):
        """
        notes : (onset idx, offset idx, pitch, velocity)
        """

        def _add_eos(tokens):
            tokens = np.concatenate((tokens, np.array([EOS], dtype=tokens.dtype)))
            return tokens

        def _add_composer(tokens, composer_value):
            tokens = np.concatenate(
                (np.array([composer_value], dtype=tokens.dtype), tokens)
            )
            return tokens

        if len(notes) == 0:
            tokens = np.array([], dtype=int)
            if add_eos:
                tokens = _add_eos(tokens)
            if add_composer:
                tokens = _add_composer(tokens, composer_value=composer_value)
            return tokens

        max_time_idx = notes[:, :2].max()

        # times[time_idx] = [[pitch, .. ], [pitch, 0], ..]
        times = [[] for i in range((max_time_idx + 1 - offset_idx))]
        for abs_onset, abs_offset, pitch, velocity in notes:
            rel_onset = abs_onset - offset_idx
            rel_offset = abs_offset - offset_idx
            times[rel_onset].append([pitch, velocity])
            times[rel_offset].append([pitch, 0])

        # 여기서부터는 전부 시간 0(offset) 기준
        tokens = []
        current_velocity = 0
        current_time_idx = 0

        for rel_idx, time in enumerate(times):
            if len(time) == 0:
                continue
            time_idx_shift = rel_idx - current_time_idx
            current_time_idx = rel_idx

            tokens.append(self.tokenize_note(time_idx_shift, TOKEN_TIME))
            for pitch, velocity in time:
                velocity = int(velocity > 0)
                if current_velocity != velocity:
                    current_velocity = velocity
                    tokens.append(self.tokenize_note(velocity, TOKEN_VELOCITY))
                tokens.append(self.tokenize_note(pitch, TOKEN_NOTE))

        tokens = np.array(tokens, dtype=int)
        if add_eos:
            tokens = _add_eos(tokens)
        if add_composer:
            tokens = _add_composer(tokens, composer_value=composer_value)
        return tokens

    def relative_batch_tokens_to_midi(
        self,
        tokens,
        beatstep,
        beat_offset_idx=None,
        bars_per_batch=None,
        cutoff_time_idx=None,
        midi_bpm=120.0,
    ):
        """
        tokens : (batch, sequence)
        beatstep : (times, )
        """
        beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
        notes = None
        bars_per_batch = 2 if bars_per_batch is None else bars_per_batch

        N = len(tokens)
        for n in range(N):
            _tokens = tokens[n]
            _start_idx = beat_offset_idx + n * bars_per_batch * 4
            _cutoff_time_idx = cutoff_time_idx + _start_idx
            _notes = self.relative_tokens_to_notes(
                _tokens,
                start_idx=_start_idx,
                cutoff_time_idx=_cutoff_time_idx,
            )
            # print(_notes, "\n-------")
            if len(_notes) == 0:
                pass
                # print("_notes zero")
            elif notes is None:
                notes = _notes
            else:
                notes = np.concatenate((notes, _notes), axis=0)

        if notes is None:
            notes = []
        midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx], midi_bpm=midi_bpm)
        return midi, notes

    def relative_tokens_to_notes(self, tokens, start_idx, cutoff_time_idx=None):
        # TODO remove legacy
        # decoding 첫토큰이 편곡자인 경우
        if tokens[0] >= sum(self.config.vocab_size.values()):
            tokens = tokens[1:]

        words = [self.detokenize(token, time_idx_offset=0) for token in tokens]

        if hasattr(start_idx, "item"):
            """
            if numpy or torch tensor
            """
            start_idx = start_idx.item()

        current_idx = start_idx
        current_velocity = 0
        note_onsets_ready = [None for i in range(self.config.vocab_size.note + 1)]
        notes = []
        for type, number in words:
            if type == TOKEN_SPECIAL:
                if number == EOS:
                    break
            elif type == TOKEN_TIME:
                current_idx += number
                if cutoff_time_idx is not None:
                    current_idx = min(current_idx, cutoff_time_idx)

            elif type == TOKEN_VELOCITY:
                current_velocity = number
            elif type == TOKEN_NOTE:
                pitch = number
                if current_velocity == 0:
                    # note_offset
                    if note_onsets_ready[pitch] is None:
                        # offset without onset
                        pass
                    else:
                        onset_idx = note_onsets_ready[pitch]
                        if onset_idx >= current_idx:
                            # No time shift after previous note_on
                            pass
                        else:
                            offset_idx = current_idx
                            notes.append(
                                [onset_idx, offset_idx, pitch, DEFAULT_VELOCITY]
                            )
                            note_onsets_ready[pitch] = None
                else:
                    # note_on
                    if note_onsets_ready[pitch] is None:
                        note_onsets_ready[pitch] = current_idx
                    else:
                        # note-on already exists
                        onset_idx = note_onsets_ready[pitch]
                        if onset_idx >= current_idx:
                            # No time shift after previous note_on
                            pass
                        else:
                            offset_idx = current_idx
                            notes.append(
                                [onset_idx, offset_idx, pitch, DEFAULT_VELOCITY]
                            )
                            note_onsets_ready[pitch] = current_idx
            else:
                raise ValueError

        for pitch, note_on in enumerate(note_onsets_ready):
            # force offset if no offset for each pitch
            if note_on is not None:
                if cutoff_time_idx is None:
                    cutoff = note_on + 1
                else:
                    cutoff = max(cutoff_time_idx, note_on + 1)

                offset_idx = max(current_idx, cutoff)
                notes.append([note_on, offset_idx, pitch, DEFAULT_VELOCITY])

        if len(notes) == 0:
            return []
        else:
            notes = np.array(notes)
            note_order = notes[:, 0] * 128 + notes[:, 1]
            notes = notes[note_order.argsort()]
            return notes

    def notes_to_midi(self, notes, beatstep, offset_sec=None, midi_bpm=120.0):
        if midi_bpm <= 0.0:
            midi_bpm = 120.0
        new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=midi_bpm)
        new_inst = pretty_midi.Instrument(program=0)
        new_notes = []
        if offset_sec is None:
            offset_sec = 0.0

        for onset_idx, offset_idx, pitch, velocity in notes:
            new_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=beatstep[onset_idx] - offset_sec,
                end=beatstep[offset_idx] - offset_sec,
            )
            new_notes.append(new_note)
        new_inst.notes = new_notes
        new_pm.instruments.append(new_inst)
        new_pm.remove_invalid_notes()
        return new_pm


@jit(nopython=True, cache=False)
def fast_notes_to_relative_tokens(
    notes, offset_idx, max_time_idx, n_special, n_note, n_velocity
):
    """
    notes : (onset idx, offset idx, pitch, velocity)
    """

    times_p = [np.array([], dtype=int) for i in range((max_time_idx + 1 - offset_idx))]
    times_v = [np.array([], dtype=int) for i in range((max_time_idx + 1 - offset_idx))]

    for abs_onset, abs_offset, pitch, velocity in notes:
        rel_onset = abs_onset - offset_idx
        rel_offset = abs_offset - offset_idx
        times_p[rel_onset] = np.append(times_p[rel_onset], pitch)
        times_v[rel_onset] = np.append(times_v[rel_onset], velocity)
        times_p[rel_offset] = np.append(times_p[rel_offset], pitch)
        times_v[rel_offset] = np.append(times_v[rel_offset], velocity)

    # 여기서부터는 전부 시간 0(offset) 기준
    tokens = []
    current_velocity = np.array([0])
    current_time_idx = np.array([0])

    # range가 0일 수도 있으니까..
    for i in range(len(times_p)):
        rel_idx = i
        notes_at_time = times_p[i]
        if len(notes_at_time) == 0:
            continue

        time_idx_shift = rel_idx - current_time_idx[0]
        current_time_idx[0] = rel_idx

        token = fast_tokenize(
            time_idx_shift,
            TOKEN_TIME,
            n_special=n_special,
            n_note=n_note,
            n_velocity=n_velocity,
        )
        tokens.append(token)

        for j in range(len(notes_at_time)):
            pitch = times_p[j]
            velocity = times_v[j]
            # for pitch, velocity in time:
            velocity = int(velocity > 0)
            if current_velocity[0] != velocity:
                current_velocity[0] = velocity
                token = fast_tokenize(
                    velocity,
                    TOKEN_VELOCITY,
                    n_special=n_special,
                    n_note=n_note,
                    n_velocity=n_velocity,
                )
                tokens.append(token)
            token = fast_tokenize(
                pitch,
                TOKEN_NOTE,
                n_special=n_special,
                n_note=n_note,
                n_velocity=n_velocity,
            )
            tokens.append(token)

    return np.array(tokens)
