import copy
import librosa
import essentia
import essentia.standard
import numpy as np
import scipy.interpolate as interp
import note_seq

SAMPLERATE = 44100


def nearest_onset_offset_digitize(on, off, bins):
    intermediate = (bins[1:] + bins[:-1]) / 2
    on_idx = np.digitize(on, intermediate)
    off_idx = np.digitize(off, intermediate)
    off_idx[on_idx == off_idx] += 1
    # off_idx = np.clip(off_idx, a_min=0, a_max=len(bins) - 1)
    return on_idx, off_idx


def apply_sustain_pedal(pm):
    ns = note_seq.midi_to_note_sequence(pm)
    susns = note_seq.apply_sustain_control_changes(ns)
    suspm = note_seq.note_sequence_to_pretty_midi(susns)
    return suspm


def interpolate_beat_times(beat_times, steps_per_beat, extend=False):
    beat_times_function = interp.interp1d(
        np.arange(beat_times.size),
        beat_times,
        bounds_error=False,
        fill_value="extrapolate",
    )
    if extend:
        beat_steps_8th = beat_times_function(
            np.linspace(0, beat_times.size, beat_times.size * steps_per_beat + 1)
        )
    else:
        beat_steps_8th = beat_times_function(
            np.linspace(0, beat_times.size - 1, beat_times.size * steps_per_beat - 1)
        )
    return beat_steps_8th


def midi_quantize_by_beats(
    sample, beat_times, steps_per_beat, ignore_sustain_pedal=False
):
    ns = note_seq.midi_file_to_note_sequence(sample.midi)
    if ignore_sustain_pedal:
        susns = ns
    else:
        susns = note_seq.apply_sustain_control_changes(ns)

    qns = copy.deepcopy(susns)

    notes = np.array([[n.start_time, n.end_time] for n in susns.notes])
    note_attributes = np.array([[n.pitch, n.velocity] for n in susns.notes])

    note_ons = np.array(notes[:, 0])
    note_offs = np.array(notes[:, 1])

    beat_steps_8th = interpolate_beat_times(beat_times, steps_per_beat, extend=False)

    on_idx, off_idx = nearest_onset_offset_digitize(note_ons, note_offs, beat_steps_8th)

    beat_steps_8th = interpolate_beat_times(beat_times, steps_per_beat, extend=True)

    discrete_notes = np.concatenate(
        (np.stack((on_idx, off_idx), axis=1), note_attributes), axis=1
    )

    def delete_duplicate_notes(dnotes):
        note_order = dnotes[:, 0] * 128 + dnotes[:, 2]
        dnotes = dnotes[note_order.argsort()]
        indices = []
        for i in range(1, len(dnotes)):
            if dnotes[i, 0] == dnotes[i - 1, 0] and dnotes[i, 2] == dnotes[i - 1, 2]:
                indices.append(i)
        dnotes = np.delete(dnotes, indices, axis=0)
        note_order = dnotes[:, 0] * 128 + dnotes[:, 1]
        dnotes = dnotes[note_order.argsort()]
        return dnotes

    discrete_notes = delete_duplicate_notes(discrete_notes)

    digitized_note_ons, digitized_note_offs = (
        beat_steps_8th[on_idx],
        beat_steps_8th[off_idx],
    )

    for i, note in enumerate(qns.notes):
        note.start_time = digitized_note_ons[i]
        note.end_time = digitized_note_offs[i]

    return qns, discrete_notes, beat_steps_8th


def extract_rhythm(song, y=None):
    if y is None:
        y, sr = librosa.load(song, sr=SAMPLERATE)

    essentia_tracker = essentia.standard.RhythmExtractor2013(method="multifeature")
    (
        bpm,
        beat_times,
        confidence,
        estimates,
        essentia_beat_intervals,
    ) = essentia_tracker(y)
    return bpm, beat_times, confidence, estimates, essentia_beat_intervals
