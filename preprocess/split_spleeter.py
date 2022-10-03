import glob
import os
import random
import sys

from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midiaudiopair import MidiAudioPair


def split_spleeter(meta_files):
    # Use audio loader explicitly for loading audio waveform :
    from spleeter.audio.adapter import AudioAdapter
    from spleeter.separator import Separator
    import spleeter

    sample_rate = 44100
    audio_loader = AudioAdapter.default()

    # Using embedded configuration.
    separator = Separator("spleeter:2stems")

    for meta_file in tqdm(meta_files):
        sample = MidiAudioPair(meta_file)
        if sample.error_code == MidiAudioPair.NO_SONG:
            continue
        if os.path.exists(sample.vocals):
            continue

        waveform, _ = audio_loader.load(sample.song, sample_rate=sample_rate)

        # Perform the separation :
        prediction = separator.separate(waveform)

        audio_loader.save(
            path=sample.vocals,
            data=prediction["vocals"][:, 0:1],
            codec=spleeter.audio.Codec.MP3,
            sample_rate=sample_rate,
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
        "--random_order",
        default=False,
        action="store_true",
        help="Random order process (to run multiple process)",
    )

    args = parser.parse_args()

    meta_files = sorted(glob.glob(args.data_dir + "/*.yaml"))
    if args.random_order:
        random.shuffle(meta_files)

    print("meta ", len(meta_files))

    split_spleeter(meta_files)
