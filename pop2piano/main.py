import os
import sys
from configargparse import ArgumentParser

import torch
from omegaconf import OmegaConf
import note_seq
from .transformer_wrapper import TransformerWrapper


DEFAULT_CONFIG_FILES = ['./.pop2piano', '~/.pop2piano']
AUDIO_EXTENSIONS = ["wav", "mp3"]
COMPOSERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
               "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"
]


def main_arguments_parser():
    def str2bool(v):
        return str(v).lower() in ('yes', 'true', 't', 'y', '1')

    parser = ArgumentParser(
        description='pop2piano: generate a piano cover of any pop song.',
        default_config_files=DEFAULT_CONFIG_FILES
    )

    parser.add('-C', '--config', is_config_file=True, help='pop2piano config file path.\nUse it to store yaml paths.')
    parser.add('-y', '--yaml', help='The yaml config file path.')
    parser.add('-M', '--model', help='The model checkpoint (.ckpt) file path.')
    parser.add_argument('--output', '-o', type=str, nargs='?',
                        help='The directory for output files. Defaults to none, i.e. output is put in input folder.')

    parser.add('-c', '--composers',
               help='The arrangement composer.'
                    '\nShould be an integer between 1 and 21.'
                    '\nMultiple composers can be specified by putting commas between them, e.g. -c 1,3,20'
                    '\nIt is also possible to select all with -c all.'
    )
    parser.add_argument('--midi', '-m', type=str2bool, nargs='?',
                        default=True, help='Whether to save mix file. Defaults to true.')
    parser.add_argument('--mix', '-x', type=str2bool, nargs='?',
                        default=True, help='Whether to save mix file. Defaults to true.')
    parser.add_argument('--plot', '-p', type=str2bool, nargs='?',
                        default=False, help='Whether to save show the plot. Defaults to false.')
    parser.add_argument('--nocuda', '-n', action="store_true", help='Pass this flag to run on CPU (without CUDA)')

    parser.add_argument('input', nargs='+',
                        default=False, help='Input files or folders. Accepts mp3 and wav files.')

    return parser


def parse_composer(arg):
    candidates = (arg or "").strip().split(",")
    if "all" in candidates:
        composers = COMPOSERS
    else:
        composers = [c for c in candidates if c in COMPOSERS] or ["1"]
    return ["composer" + c for c in composers]


def parse_input(args):
    input_files = []
    for arg in args:
        if os.path.isdir(arg):
            for root, dirs, files in os.walk(arg):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                        input_files.append(os.path.join(root, f))
        else:
            if any(arg.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                input_files.append(arg)
    return input_files


def main():
    args = main_arguments_parser().parse_args()
    input_files = parse_input(args.input)
    composers = parse_composer(args.composers)

    config_path = args.yaml or "~/src/pop2piano/config.yaml"
    config_path = os.path.realpath(os.path.expanduser(config_path))
    model_path = args.model or "~/src/pop2piano/model-1999-val_0.67311615.ckpt"
    model_path = os.path.realpath(os.path.expanduser(model_path))

    device = "cuda"
    if not torch.cuda.is_available():
        if not args.nocuda:
            print("CUDA unavailable. "
                  "If it should be, you can restart it with the following command:"
                  "sudo rmmod nvidia_uvm; sudo modprobe nvidia_uvm"
                  "If not, run the command again with the nocuda flag.")
            sys.exit(1)
        device = "cpu"
    config = OmegaConf.load(config_path)
    wrapper = TransformerWrapper(config)
    wrapper = wrapper.load_from_checkpoint(model_path, config=config).to(device)
    wrapper.eval()

    for audio_file in input_files:
        for composer in composers:
            try:
                pm, composer, mix_path, midi_path = wrapper.generate(
                    audio_path=audio_file,
                    composer=composer,
                    model="dpipqxiy",
                    show_plot=args.plot,
                    save_midi=args.midi,
                    save_mix=args.mix,
                    output_prefix=args.output,
                )
                # note_seq.plot_sequence(note_seq.midi_to_note_sequence(pm))
            except:
                print(f"Error: could not process files {audio_file} for {composer}")

if __name__ == "__main__":
    main()
