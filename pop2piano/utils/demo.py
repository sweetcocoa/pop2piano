import re
import os

import shutil
import youtube_dl
from youtube_dl.utils import sanitize_filename


def download_youtube(url, dst_dir, dst_filename=None, keep_video=False):
    ydl_opts = {
        "format": "mp4",
        "restrictfilenames": True,
        "keepvideo": keep_video,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        rt = ydl.extract_info(url)

    title = sanitize_filename(rt["title"], restricted=True)
    reg_title = re.sub("[^a-zA-Zㄱ-ㅎ가-힣0-9\ \-\_\.]", "", rt["title"])

    result_video_filename = f"{title}-{rt['id']}.{rt['ext']}"
    result_audio_filename = f"{title}-{rt['id']}.mp3"
    result_ok = os.path.exists(result_audio_filename)

    dst_audio_filename = (
        f"{reg_title}-{rt['id']}.mp3" if dst_filename is None else f"{dst_filename}.mp3"
    )
    dst_audio_filepath = os.path.join(dst_dir, dst_audio_filename)
    dst_video_filename = f"{reg_title}-{rt['id']}.{rt['ext']}"
    dst_video_filepath = os.path.join(dst_dir, dst_video_filename)

    if result_ok:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(result_audio_filename, dst_audio_filepath)
        if keep_video:
            shutil.move(result_video_filename, dst_video_filepath)
            return dst_audio_filepath, dst_video_filepath

    return dst_audio_filepath
