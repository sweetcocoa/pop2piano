"""
Usage:
python youtube_down.py piano_covers.txt /output/dir
"""

import os
import multiprocessing

import tempfile
import shutil
import glob
import pandas as pd
import re

from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import OmegaConf


def download_piano(
    url: str,
    output_dir: str,
    postprocess=True,
    dry_run=False,
) -> int:
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = f"{tmpdir}/%(uploader)s___%(title)s___%(id)s___%(duration)d.%(ext)s"

        if postprocess:
            postprocess_call = '--postprocessor-args "-ac 1 -ar 16000"'
        else:
            postprocess_call = ""
        result = os.system(
            f"""youtube-dl -o "{output}" \\
                --extract-audio \\
                --audio-quality 0 \\
                --audio-format wav \\
                --retries 50 \\
                --prefer-ffmpeg \\
                {"--get-filename" if dry_run else ""}\\
                {postprocess_call} \\
                --force-ipv4 \\
                --yes-playlist \\
                --ignore-errors \\
                {url}"""
        )

        if not dry_run:

            files = os.listdir(tmpdir)

            for filename in files:
                filename_wo_ext, ext = os.path.splitext(filename)
                uploader, title, ytid, duration = filename_wo_ext.split("___")
                meta = OmegaConf.create()
                meta.piano = OmegaConf.create()
                meta.piano.uploader = uploader
                meta.piano.title = title
                meta.piano.ytid = ytid
                meta.piano.duration = int(duration)
                OmegaConf.save(meta, os.path.join(output_dir, ytid + ".yaml"))
                shutil.move(
                    os.path.join(tmpdir, filename),
                    os.path.join(output_dir, f"{ytid}{ext}"),
                )

    return result


def download_piano_main(piano_list, output_dir, dry_run=False):
    """
    piano_list : list of youtube id
    """
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(download_piano)(
            url=f"https://www.youtube.com/watch?v={ytid}",
            output_dir=output_dir,
            postprocess=True,
            dry_run=dry_run,
        )
        for ytid in tqdm(piano_list)
    )


def download_pop(piano_id, pop_id, output_dir, dry_run):
    output_file_template = "%(id)s___%(title)s___%(duration)d.%(ext)s"
    pop_output_dir = os.path.join(output_dir, piano_id)
    os.makedirs(pop_output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, piano_id, output_file_template)
    url = f"https://www.youtube.com/watch?v={pop_id}"

    result = os.system(
        f"""youtube-dl -o "{output_template}" \\
            --extract-audio \\
            --audio-quality 0 \\
            --audio-format wav \\
            --retries 25 \\
            {"--get-filename" if dry_run else ""}\\
            --prefer-ffmpeg \\
            --match-filter 'duration < 300 & duration > 150'\\
            --postprocessor-args "-ac 2 -ar 44100" \\
            {url}"""
    )

    if not dry_run:
        files = list(filter(lambda x: x.endswith(".wav"), os.listdir(pop_output_dir)))
        files = glob.glob(os.path.join(pop_output_dir, "*.wav"))
        for filename in files:
            filename_wo_ext, ext = os.path.splitext(os.path.basename(filename))
            ytid, title, duration = filename_wo_ext.split("___")
            yaml = os.path.join(output_dir, piano_id + ".yaml")

            meta = OmegaConf.load(yaml)
            meta.song = OmegaConf.create()
            meta.song.ytid = ytid
            meta.song.title = title
            meta.song.duration = int(duration)

            OmegaConf.save(meta, yaml)
            shutil.move(
                os.path.join(filename),
                os.path.join(output_dir, f"{ytid}{ext}"),
            )


def download_pop_main(piano_list, pop_list, output_dir, dry_run=False):
    """
    piano_list : list of youtube id
    pop_list : corresponding youtube id of pop songs
    """

    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(download_pop)(
            piano_id=piano_id,
            pop_id=pop_id,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        for piano_id, pop_id in tqdm(list(zip(piano_list, pop_list)))
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="piano cover downloader")

    parser.add_argument("dataset", type=str, default=None, help="provided csv")
    parser.add_argument("output_dir", type=str, default=None, help="output dir")
    parser.add_argument("--dry_run", default=False, action="store_true", help="whether dry_run")

    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[:50]

    piano_list = df["piano_ids"].tolist()
    # download_piano_main(piano_list, args.output_dir, args.dry_run)

    available_piano_list = glob.glob(args.output_dir + "/**/*.yaml", recursive=True)
    df.index = df["piano_ids"]

    failed_piano = []

    available_piano_list_id = [
        os.path.splitext(os.path.basename(ap))[0] for ap in available_piano_list
    ]

    for piano_id_to_be_downloaded in tqdm(df["piano_ids"]):
        if piano_id_to_be_downloaded in available_piano_list_id:
            continue
        else:
            failed_piano.append(piano_id_to_be_downloaded)

    if len(failed_piano) > 0:
        print(f"{len(failed_piano)} of files are failed to be downloaded")
        df = df.drop(index=failed_piano)

    piano_list = df["piano_ids"].tolist()
    pop_list = df["pop_ids"].tolist()

    download_pop_main(piano_list, pop_list, output_dir=args.output_dir, dry_run=args.dry_run)
