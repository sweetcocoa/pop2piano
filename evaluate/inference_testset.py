import os
import sys
import glob

import librosa
import torch
import numpy as np
import pretty_midi
from omegaconf import OmegaConf
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midiaudiopair import MidiAudioPair
from transformer_wrapper import TransformerWrapper, DEFAULT_COMPOSERS
from evaluate import midi_melody_accuracy as ma
from sweetdebug import sweetdebug


def inference_main(meta_files, ckpt, config, id, **kwargs):

    import warnings

    sweetdebug(use_telegram_if_cache_exists=False)
    warnings.filterwarnings(action="ignore")

    config = OmegaConf.load(config)
    wrapper = TransformerWrapper(config)
    wrapper = wrapper.load_from_checkpoint(ckpt, config=config).cuda()
    wrapper.eval()

    with torch.no_grad():
        for meta_file in tqdm(meta_files):
            sample = MidiAudioPair(meta_file)

            # Pass if the midi of all composers are generated.
            # -------------------------------------------
            some_not_generated = False
            for composer, value in wrapper.composer_to_feature_token.items():
                midi_path = sample.generated(composer=composer, generated=id)
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)

                if not os.path.exists(midi_path):
                    some_not_generated = True

            all_generated = not some_not_generated
            if all_generated:
                continue
            # ---------------------------------------------

            # load pre-computed beats
            # ------------------------------------
            beatstep = np.load(sample.beatstep)
            # ------------------------------------

            # load audio if needed
            if wrapper.use_mel:
                y, sr = librosa.load(sample.song, sr=config.dataset.sample_rate)
                vqvae_token = None
            else:
                vqvae_token = torch.load(sample.vqvae, map_location="cuda")
                y = None
                sr = None

            for composer, value in wrapper.composer_to_feature_token.items():
                midi_path = sample.generated(composer=composer, generated=id)
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)

                if os.path.exists(midi_path):
                    continue

                wrapper.generate(
                    audio_path=None,
                    composer=composer,
                    model=id,
                    save_midi=True,
                    save_mix=False,
                    show_plot=False,
                    midi_path=midi_path,
                    vqvae_token=vqvae_token,
                    beatsteps=beatstep - beatstep[0],
                    audio_y=y,
                    audio_sr=sr,
                )


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


def evaluate_main(meta_files, config, model_id, **kwargs):
    from tqdm.auto import tqdm
    import multiprocessing
    from joblib import Parallel, delayed

    config = OmegaConf.load(config)
    composer_dic = config.composer_to_feature_token
    if config.dataset.use_mel and not config.dataset.mel_is_conditioned:
        composer_dic = DEFAULT_COMPOSERS

    mean_accuracys = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(evaluate)(meta_file, composer_dic, model_id)
        for meta_file in tqdm(meta_files)
    )

    print(
        "Total Accuracy of", model_id, "is", sum(mean_accuracys) / len(mean_accuracys)
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
        "--ckpt",
        type=str,
        default=None,
        help="""ckpt *.ckpt""",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="""config *.yaml""",
    )

    parser.add_argument(
        "--id",
        type=str,
        default="model_name_id",
        help="""config composer_to_token.yaml""",
    )

    parser.add_argument("--evaluate", action="store_true", default=False)

    args = parser.parse_args()

    meta_files = sorted(glob.glob(args.data_dir + "/**/*.yaml", recursive=True))
    print("meta ", len(meta_files))

    inference_main(meta_files, **vars(args))
    if args.evaluate:
        evaluate_main(meta_files, config=args.config, model_id=args.id)
