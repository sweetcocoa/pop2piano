import torch, os, glob
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio


def transcribe_to_midi(
    wav_input_dir: str,
    midi_output_dir: str,
):
    midi_output_dir = os.path.realpath(midi_output_dir)
    print(midi_output_dir)
    print(os.path.dirname(midi_output_dir))
    os.makedirs(midi_output_dir, exist_ok=True)

    filenames = list(filter(lambda x: x.endswith(".wav"), os.listdir(wav_input_dir)))
    print(f"{filenames=}")
    files = glob.glob(os.path.join(wav_input_dir, "*.wav"))
    for filename in files:
        (audio, _) = load_audio(filename, sr=sample_rate, mono=True)
        transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)  # device: 'cuda' | 'cpu'
        outfile = os.path.join(midi_output_dir, os.path.splitext(os.path.basename(filename))[0])+'.mid'
        print(outfile)
        transcribed_dict = transcriptor.transcribe(audio, outfile)
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="piano to midi transcriber")

    parser.add_argument("input_dir", type=str, default=None, help="provided wavs")
    parser.add_argument("output_dir", type=str, default=None, help="output dir for midis")
    args = parser.parse_args()

    transcribe_to_midi(args.input_dir, args.output_dir)