import glob
import sys
import os

import librosa
import pretty_midi

from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midiaudiopair import MidiAudioPair
from evaluate import midi_melody_accuracy as ma


def estimate(meta_file):

    import warnings

    warnings.filterwarnings(action="ignore")

    sample = MidiAudioPair(meta_file)

    if (
        sample.error_code == MidiAudioPair.NO_PIANO
        or sample.error_code == MidiAudioPair.NO_SONG_DIR
        or sample.error_code == MidiAudioPair.NO_SONG
    ):
        return

    if "vocals" in sample.invalids:
        print("no vocal:", meta_file)
        return

    midi = pretty_midi.PrettyMIDI(sample.qmidi)
    vocals, sr = librosa.load(sample.vocals, sr=44100)

    chroma_accuracy, pitch_accuracy = ma.evaluate_melody(
        midi, vocals, sr=sr, hop_length=1024
    )
    meta = OmegaConf.load(meta_file)
    meta.eval = OmegaConf.create()
    meta.eval.melody_chroma_accuracy = chroma_accuracy.item()
    meta.eval.melody_pitch_accuracy = pitch_accuracy.item()
    OmegaConf.save(meta, meta_file)


def main(meta_files):
    from tqdm import tqdm
    import multiprocessing
    from joblib import Parallel, delayed

    def files():
        pbar = tqdm(meta_files)
        for meta_file in pbar:
            pbar.set_description(meta_file)
            yield meta_file

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(estimate)(meta_file) for meta_file in files()
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

    args = parser.parse_args()

    meta_files = sorted(glob.glob(args.data_dir + "/**/*.yaml", recursive=True))
    print("meta ", len(meta_files))

    main(meta_files)
