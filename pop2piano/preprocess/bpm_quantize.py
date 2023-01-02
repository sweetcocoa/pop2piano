import glob
import sys
import os


import librosa
import soundfile as sf
import numpy as np

import note_seq
from omegaconf import OmegaConf
from beat_quantizer import extract_rhythm, midi_quantize_by_beats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..midiaudiopair import MidiAudioPair
from ..utils.dsp import get_stereo


def estimate(meta_file, ignore_sustain_pedal):
    sample = MidiAudioPair(meta_file)

    if (
        sample.error_code == MidiAudioPair.NO_PIANO
        or sample.error_code == MidiAudioPair.NO_SONG_DIR
        or sample.error_code == MidiAudioPair.NO_SONG
    ):
        return

    bpm, beat_times, confidence, estimates, essentia_beat_intervals = extract_rhythm(
        sample.song
    )
    beat_times = np.array(beat_times)
    essentia_beat_intervals = np.array(essentia_beat_intervals)

    qns, discrete_notes, beat_steps_8th = midi_quantize_by_beats(
        sample, beat_times, 2, ignore_sustain_pedal=ignore_sustain_pedal
    )

    qpm = note_seq.note_sequence_to_pretty_midi(qns)
    qpm.instruments[0].control_changes = []
    qpm.write(sample.qmidi)
    y, sr = librosa.load(sample.song, sr=None)
    qpm_y = qpm.fluidsynth(sr)
    qmix = get_stereo(y, qpm_y, 0.4)
    sf.write(file=sample.qmix, data=qmix.T, samplerate=sr, format="flac")

    meta = OmegaConf.load(meta_file)
    meta.tempo = OmegaConf.create()
    meta.tempo.bpm = bpm
    meta.tempo.confidence = confidence
    OmegaConf.save(meta, meta_file)

    np.save(sample.notes, discrete_notes)
    np.save(sample.beatstep, beat_steps_8th)
    np.save(sample.beattime, beat_times)
    np.save(sample.beatinterval, essentia_beat_intervals)


def main(meta_files, ignore_sustain_pedal):
    from tqdm import tqdm
    import multiprocessing
    from joblib import Parallel, delayed

    def files():
        pbar = tqdm(meta_files)
        for meta_file in pbar:
            pbar.set_description(meta_file)
            yield meta_file

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(estimate)(meta_file, ignore_sustain_pedal) for meta_file in files()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="bpm estimate using essentia")

    parser.add_argument(
        "data_dir",
        type=str,
        default=None,
        help="""directory contains {id}/{pop_filename.wav}
        """,
    )

    parser.add_argument(
        "--ignore_sustain_pedal",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    meta_files = sorted(glob.glob(args.data_dir + "/*.yaml"))
    print("meta ", len(meta_files))

    main(meta_files, args.ignore_sustain_pedal)
