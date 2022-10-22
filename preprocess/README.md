# Preprocess Scripts
---
- Note : the order of these scripts is IMPORTANT. 
- the preprocessing step is easy. but environment setting is not. please understand.
- If you encounter any problems, please do not hesitate to email me or open an issue to the github.

1. Transcribe piano wavs to midi
- You should transcribe {piano_cover_file.wav} -> {piano_cover_file.mid}
- I recommend you to use original codes from this repo : [High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times](https://github.com/qiuqiangkong/piano_transcription_inference)

- Instead, you can also you my docker script.
    ```bash
    docker run -it --gpus all --rm -v /DIRECTORY_THAT_CONTAINS_PIANO_WAV/:/input -v /DIRECTORY_THAT_MIDI_OUTPUT/:/output jonghochoi/piano_transcribe:bytedance1
    ```
- If you are using GPU RTX 30XX or higher, this script may not work properly. It's because the version of pytorch is too low(1.4).
- then upgrade the version of pytorch in the docker..

2. Estimate Pop's beats
```bash
python bpm_quantize.py DATA_DIR 
```

3. synchronize midi 
```bash
python pop_align.py DATA_DIR
```

4. get separated vocal track
```bash
python split_spleeter.py DATA_DIR
```

5. caculate melody chroma accuracy
```bash
python melody_accuracy.py DATA_DIR 
```