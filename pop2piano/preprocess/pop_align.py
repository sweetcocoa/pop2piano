import librosa
import soundfile as sf
import glob
import os
import copy
import sys

import numpy as np
import pyrubberband as pyrb
import pretty_midi
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import (
    compute_optimal_chroma_shift,
    shift_chroma_vectors,
    make_path_strictly_monotonic,
)
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dsp import normalize, get_stereo
from midiaudiopair import MidiAudioPair

Fs = 22050
feature_rate = 50
step_weights = np.array([1.5, 1.5, 2.0])
threshold_rec = 10 ** 6


def save_delayed_song(
    sample,
    dry_run,
):
    import warnings

    warnings.filterwarnings(action="ignore")

    song_audio, _ = librosa.load(sample.original_song, Fs)
    midi_pm = pretty_midi.PrettyMIDI(sample.original_midi)

    if np.power(song_audio, 2).sum() < 1:  # low energy: invalid file
        print("invalid audio :", sample.original_song)
        sample.delete_files_myself()
        return

    rd = get_aligned_results(midi_pm=midi_pm, song_audio=song_audio)

    mix_song = rd["mix_song"]
    song_pitch_shifted = rd["song_pitch_shifted"]
    midi_warped_pm = rd["midi_warped_pm"]
    pitch_shift_for_song_audio = rd["pitch_shift_for_song_audio"]
    tuning_offset_song = rd["tuning_offset_song"]
    tuning_offset_piano = rd["tuning_offset_piano"]

    try:
        if dry_run:
            print("write audio files: ", sample.song)
        else:
            sf.write(
                file=sample.song,
                data=song_pitch_shifted,
                samplerate=Fs,
                format="wav",
            )
    except:
        print("Fail : ", sample.song)

    try:
        if dry_run:
            print("write warped midi :", sample.midi)
        else:
            midi_warped_pm.write(sample.midi)

    except:
        midi_warped_pm._tick_scales = midi_pm._tick_scales
        try:
            if dry_run:
                print("write warped midi2 :", sample.midi)
            else:
                midi_warped_pm.write(sample.midi)

        except:
            print("ad-hoc failed midi : ", sample.midi)
        print("ad-hoc midi : ", sample.midi)

    sample.yaml.song.pitch_shift = pitch_shift_for_song_audio.item()
    sample.yaml.song.tuning_offset = tuning_offset_song.item()
    sample.yaml.piano.tuning_offset = tuning_offset_piano.item()
    OmegaConf.save(sample.yaml, sample.yaml_path)


def get_aligned_results(midi_pm, song_audio):
    piano_audio = midi_pm.fluidsynth(Fs)

    song_audio = normalize(song_audio)

    # The reason for estimating tuning ::
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_TranspositionTuning.html
    tuning_offset_1 = estimate_tuning(song_audio, Fs)
    tuning_offset_2 = estimate_tuning(piano_audio, Fs)

    # DLNCO features (Sebastian Ewert, Meinard Müller, and Peter Grosche: High Resolution Audio Synchronization Using Chroma Onset Features, In Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP): 1869–1872, 2009.):
    # helpful to increase synchronization accuracy, especially for music with clear onsets.

    # Quantized and smoothed chroma : CENS features
    # Because, MrMsDTW Requires CENS.
    f_chroma_quantized_1, f_DLNCO_1 = get_features_from_audio(
        song_audio, tuning_offset_1
    )
    f_chroma_quantized_2, f_DLNCO_2 = get_features_from_audio(
        piano_audio, tuning_offset_2
    )

    # Shift chroma vectors :
    # Otherwise, different keys of two audio leads to degradation of alignment.
    opt_chroma_shift = compute_optimal_chroma_shift(
        quantized_chroma_to_CENS(f_chroma_quantized_1, 201, 50, feature_rate)[0],
        quantized_chroma_to_CENS(f_chroma_quantized_2, 201, 50, feature_rate)[0],
    )
    f_chroma_quantized_2 = shift_chroma_vectors(f_chroma_quantized_2, opt_chroma_shift)
    f_DLNCO_2 = shift_chroma_vectors(f_DLNCO_2, opt_chroma_shift)

    wp = sync_via_mrmsdtw(
        f_chroma1=f_chroma_quantized_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_quantized_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=feature_rate,
        step_weights=step_weights,
        threshold_rec=threshold_rec,
        verbose=False,
    )

    wp = make_path_strictly_monotonic(wp)
    pitch_shift_for_song_audio = -opt_chroma_shift % 12
    if pitch_shift_for_song_audio > 6:
        pitch_shift_for_song_audio -= 12

    if pitch_shift_for_song_audio != 0:
        song_audio_shifted = pyrb.pitch_shift(
            song_audio, Fs, pitch_shift_for_song_audio
        )
    else:
        song_audio_shifted = song_audio

    time_map_second = wp / feature_rate
    midi_pm_warped = copy.deepcopy(midi_pm)

    midi_pm_warped = simple_adjust_times(
        midi_pm_warped, time_map_second[1], time_map_second[0]
    )
    piano_audio_warped = midi_pm_warped.fluidsynth(Fs)

    song_audio_shifted = normalize(song_audio_shifted)
    stereo_sonification_piano = get_stereo(song_audio_shifted, piano_audio_warped)

    rd = dict(
        mix_song=stereo_sonification_piano,
        song_pitch_shifted=song_audio_shifted,
        midi_warped_pm=midi_pm_warped,
        pitch_shift_for_song_audio=pitch_shift_for_song_audio,
        tuning_offset_song=tuning_offset_1,
        tuning_offset_piano=tuning_offset_2,
    )
    return rd


def simple_adjust_times(pm, original_times, new_times):
    """
    most of these codes are from original pretty_midi
    https://github.com/craffel/pretty-midi/blob/main/pretty_midi/pretty_midi.py
    """
    for instrument in pm.instruments:
        instrument.notes = [
            copy.deepcopy(note)
            for note in instrument.notes
            if note.start >= original_times[0] and note.end <= original_times[-1]
        ]
    # Get array of note-on locations and correct them
    note_ons = np.array(
        [note.start for instrument in pm.instruments for note in instrument.notes]
    )
    adjusted_note_ons = np.interp(note_ons, original_times, new_times)
    # Same for note-offs
    note_offs = np.array(
        [note.end for instrument in pm.instruments for note in instrument.notes]
    )
    adjusted_note_offs = np.interp(note_offs, original_times, new_times)
    # Correct notes
    for n, note in enumerate(
        [note for instrument in pm.instruments for note in instrument.notes]
    ):
        note.start = (adjusted_note_ons[n] > 0) * adjusted_note_ons[n]
        note.end = (adjusted_note_offs[n] > 0) * adjusted_note_offs[n]
    # After performing alignment, some notes may have an end time which is
    # on or before the start time.  Remove these!
    pm.remove_invalid_notes()

    def adjust_events(event_getter):
        """This function calls event_getter with each instrument as the
        sole argument and adjusts the events which are returned."""
        # Sort the events by time
        for instrument in pm.instruments:
            event_getter(instrument).sort(key=lambda e: e.time)
        # Correct the events by interpolating
        event_times = np.array(
            [
                event.time
                for instrument in pm.instruments
                for event in event_getter(instrument)
            ]
        )
        adjusted_event_times = np.interp(event_times, original_times, new_times)
        for n, event in enumerate(
            [
                event
                for instrument in pm.instruments
                for event in event_getter(instrument)
            ]
        ):
            event.time = adjusted_event_times[n]
        for instrument in pm.instruments:
            # We want to keep only the final event which has time ==
            # new_times[0]
            valid_events = [
                event
                for event in event_getter(instrument)
                if event.time == new_times[0]
            ]
            if valid_events:
                valid_events = valid_events[-1:]
            # Otherwise only keep events within the new set of times
            valid_events.extend(
                event
                for event in event_getter(instrument)
                if event.time > new_times[0] and event.time < new_times[-1]
            )
            event_getter(instrument)[:] = valid_events

    # Correct pitch bends and control changes
    adjust_events(lambda i: i.pitch_bends)
    adjust_events(lambda i: i.control_changes)

    return pm


def get_features_from_audio(audio, tuning_offset, visualize=False):
    f_pitch = audio_to_pitch_features(
        f_audio=audio,
        Fs=Fs,
        tuning_offset=tuning_offset,
        feature_rate=feature_rate,
        verbose=visualize,
    )
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

    f_pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize
    )
    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=f_chroma_quantized.shape[1],
        visualize=visualize,
    )
    return f_chroma_quantized, f_DLNCO


def main(samples, dry_run):
    import multiprocessing
    from joblib import Parallel, delayed

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(save_delayed_song)(sample=sample, dry_run=dry_run)
        for sample in tqdm(samples)
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="piano cover downloader")

    parser.add_argument(
        "data_dir",
        type=str,
        default=None,
        help="""directory contains {id}/{song_filename.wav}
        """,
    )
    parser.add_argument(
        "--dry_run", default=False, action="store_true", help="whether dry_run"
    )

    args = parser.parse_args()

    def getfiles():
        meta_files = sorted(glob.glob(args.data_dir + "/*.yaml"))
        print("meta ", len(meta_files))

        samples = list()
        for meta_file in tqdm(meta_files):
            m = MidiAudioPair(meta_file, auto_remove_no_song=True)
            if m.error_code != MidiAudioPair.NO_SONG:
                aux_txt = os.path.join(
                    m.audio_dir,
                    m.yaml.piano.ytid,
                    f"{m.yaml.piano.title[:50]}___{m.yaml.song.title[:50]}.txt",
                )
                with open(aux_txt, "w") as f:
                    f.write(".")
                samples.append(m)

        print(f"files available {len(samples)}")
        return samples

    samples = getfiles()
    main(samples=samples, dry_run=args.dry_run)
