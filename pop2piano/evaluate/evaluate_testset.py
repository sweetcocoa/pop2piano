import glob
import sys
import os

import librosa
import pretty_midi

from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..midiaudiopair import MidiAudioPair
import .midi_melody_accuracy as ma
from ..transformer_wrapper import DEFAULT_COMPOSERS


def evaluate(meta_file, composer_dic, model_id):

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

    vocals, sr = librosa.load(sample.vocals, sr=44100)
    HOP_LENGTH = 1024
    f0, _, _ = ma._f0(vocals, sr, hop_length=HOP_LENGTH)

    chroma_accuracys = list()

    for composer, value in composer_dic.items():
        midi_path = sample.generated(composer, model_id)
        midi = pretty_midi.PrettyMIDI(midi_path)
        chroma_accuracy, pitch_accuracy = ma._evaluate_melody(midi, f0, sr, HOP_LENGTH)
        result = sample.result_json(model_id)
        if os.path.exists(result):
            result_json = OmegaConf.load(result)
        else:
            result_json = OmegaConf.create()
        result_json[composer] = OmegaConf.create()
        result_json[composer].melody_chroma_accuracy = chroma_accuracy.item()
        result_json[composer].melody_pitch_accuracy = pitch_accuracy.item()
        OmegaConf.save(result_json, result)
        chroma_accuracys.append(chroma_accuracy)

    mean_accuracy = sum(chroma_accuracys) / len(chroma_accuracys)
    gt_accuracy = sample.yaml.eval.melody_chroma_accuracy
    print(gt_accuracy, mean_accuracy)

    return mean_accuracy


def main(meta_files, composer_config, model_id, **kwargs):
    from tqdm.auto import tqdm
    import multiprocessing
    from joblib import Parallel, delayed

    if composer_config is None:
        composer_dic = DEFAULT_COMPOSERS
    else:
        composer_dic = OmegaConf.load(composer_config)

    # for meta_file in tqdm(meta_files):
    #     evaluate(meta_file, composer_dic, model_id)

    mean_accuracys = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(evaluate)(meta_file, composer_dic, model_id)
        for meta_file in tqdm(meta_files)
    )

    print(
        "Total Accuracy of", model_id, "is", sum(mean_accuracys) / len(mean_accuracys)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="eval melody accuracy")

    parser.add_argument(
        "data_dir",
        type=str,
        default=None,
        help="""directory contains {id}/{pop_filename.wav}
        """,
    )

    parser.add_argument(
        "--composer_config",
        type=str,
        default=None,
        help="""config composer_to_token.yaml""",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="model_id",
        help="""model id""",
    )

    args = parser.parse_args()

    meta_files = sorted(glob.glob(args.data_dir + "/**/*.yaml", recursive=True))
    print("meta ", len(meta_files))

    main(meta_files, **vars(args))
