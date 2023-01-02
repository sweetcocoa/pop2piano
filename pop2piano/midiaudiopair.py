import os
import shutil

from omegaconf import OmegaConf


BLACKLIST_PIANO_YTID = [
    "cp37xi5Jbs",
    "0meKPm-75As",
    "0uN66vwQElI",
    "S5zn1FJ29GU",
    "s_npS7szUjk",
    "SCssvPlXbvc",
    "sH7ErWQut5g",
    "DYQhtNMCzsA",
    "4CaGkbWUovE",
    "SQtrlqkIl4o",
    "ykpFk4EniDk",
    "WpHke7iywS8",
]


class MidiAudioPair:
    VALID = 0

    NO_SONG = 1
    NO_PIANO = 2
    NO_SONG_DIR = 3
    BAD_DURATION = 4
    BAD_TITLE = 5
    NO_TEMPO = 6
    BLACKLIST = 7
    BAD_ACCURACY = 8

    ERROR_CODE = {
        VALID: "Valid",
        NO_SONG: "No Song",
        NO_PIANO: "No Piano",
        NO_SONG_DIR: "No Song Dir",
        BAD_DURATION: "Duration Bad",
        BAD_TITLE: "Bad Title",
        NO_TEMPO: "No Tempo",
        BLACKLIST: "Blacklist",
        BAD_ACCURACY: "Bad Accuracy",
    }

    def validate_files(self):
        attrs = [
            "midi",
            "song",
            "beattime",
            "beatstep",
            "beatinterval",
            "qmidi",
            "qmix",
            "notes",
            "vqvae",
            "vocals",
        ]

        invalids = []
        for attr in attrs:
            file = getattr(self, attr, None)
            if file is None or not os.path.exists(file):
                invalids.append(attr)

        return invalids

    def validate_yaml(self, audio_dir, yaml):
        if not hasattr(yaml, "song"):
            return MidiAudioPair.NO_SONG

        if not hasattr(yaml, "piano"):
            return MidiAudioPair.NO_PIANO

        if yaml.piano.ytid in BLACKLIST_PIANO_YTID:
            return MidiAudioPair.BLACKLIST

        song_dir = os.path.join(audio_dir, yaml.piano.ytid)
        if not os.path.exists(song_dir) or not os.path.isdir(song_dir):
            return MidiAudioPair.NO_SONG_DIR

        piano_sec = int(yaml.piano.duration)
        song_sec = int(yaml.song.duration)
        if piano_sec / song_sec > 1.2 or piano_sec / song_sec < 0.83:
            return MidiAudioPair.BAD_DURATION

        if yaml.piano.title.find("HANPPYEOM") != -1:
            return MidiAudioPair.BAD_TITLE

        if not hasattr(yaml, "tempo"):
            return MidiAudioPair.NO_TEMPO

        if not hasattr(yaml, "eval") or yaml.eval.melody_chroma_accuracy < 0.15:
            return MidiAudioPair.BAD_ACCURACY

        return MidiAudioPair.VALID

    def set_song_attrs(self):
        basename = os.path.join(self.song_dir, f"{self.yaml.song.ytid}")

        self.mix = basename + ".mix.flac"
        self.midi = basename + ".mid"
        self.song = basename + ".pitchshift.wav"
        self.beattime = basename + ".beattime.npy"
        self.beatstep = basename + ".beatstep.npy"

        self.beatinterval = basename + ".beatinterval.npy"

        self.qmidi = basename + ".qmidi.mid"
        self.qmix = basename + ".qmix.flac"
        self.notes = basename + ".notes.npy"
        self.vqvae = basename + ".vqvae.pt"
        self.vocals = basename + ".vocals.mp3"

    def delete_files_myself(self):
        shutil.rmtree(os.path.join(self.audio_dir, self.yaml.piano.ytid))
        os.remove(self.yaml_path)
        os.remove(self.original_midi)
        if os.path.exists(self.original_wav):
            os.remove(self.original_wav)

    def __init__(self, yaml_path, audio_dir=None, auto_remove_no_song=False):
        self.yaml_path = yaml_path

        self.yaml = OmegaConf.load(yaml_path)

        self.audio_dir = (
            audio_dir if audio_dir is not None else os.path.dirname(yaml_path)
        )
        self.song_dir = os.path.join(self.audio_dir, self.yaml.piano.ytid)

        self.error_code = self.validate_yaml(self.audio_dir, self.yaml)

        self.original_midi = os.path.join(self.audio_dir, f"{self.yaml.piano.ytid}.mid")
        self.original_wav = os.path.join(self.audio_dir, f"{self.yaml.piano.ytid}.wav")

        if self.error_code == MidiAudioPair.NO_SONG:
            print("no song :", yaml_path)
            if auto_remove_no_song:
                print("remove :", yaml_path)
                self.delete_files_myself()
            return
        else:
            self.set_song_attrs()

        self.invalids = self.validate_files()
        self.is_valid = (self.error_code == MidiAudioPair.VALID) and (
            len(self.invalids) == 0
        )

        if self.error_code != MidiAudioPair.NO_SONG:
            self.original_song = os.path.join(
                self.song_dir, f"{self.yaml.song.ytid}.wav"
            )
            self.title = f"{self.yaml.piano.title}___{self.yaml.song.title}"
        else:
            self.title = f"{self.yaml.piano.title}"

    def __repr__(self):
        return f"{MidiAudioPair.ERROR_CODE[self.error_code]}, inv{self.invalids}, {self.yaml_path}, {self.title}"

    def generated(self, composer, generated="model_name"):
        midi_path = os.path.join(
            self.song_dir, generated, self.yaml.song.ytid + "." + composer + ".mid"
        )
        return midi_path

    def result_json(self, generated="model_name"):
        json_path = os.path.join(
            self.song_dir, generated, self.yaml.song.ytid + ".result.json"
        )
        return json_path
