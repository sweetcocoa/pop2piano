import numpy as np
from scipy.interpolate import interp1d


def normalize(audio, min_y=-1.0, max_y=1.0, eps=1e-8):
    assert len(audio.shape) == 1
    max_y -= eps
    min_y += eps
    amax = audio.max()
    amin = audio.min()
    audio = (max_y - min_y) * (audio - amin) / (amax - amin) + min_y
    return audio


def get_stereo(pop_y, midi_y, pop_scale=0.99):
    if len(pop_y) > len(midi_y):
        midi_y = np.pad(midi_y, (0, len(pop_y) - len(midi_y)))
    elif len(pop_y) < len(midi_y):
        pop_y = np.pad(pop_y, (0, -len(pop_y) + len(midi_y)))
    stereo = np.stack((midi_y, pop_y * pop_scale))
    return stereo


def generate_variable_f0_sine_wave(f0, len_y, sr):
    """
    integrate instant frequencies to get pure tone sine wave
    """
    x_sample = np.arange(len(f0))
    intp = interp1d(x_sample, f0, kind="linear")
    f0_audiorate = intp(np.linspace(0, len(f0) - 1, len_y))
    pitch_wave = np.sin((np.nan_to_num(f0_audiorate) / sr * 2 * np.pi).cumsum())
    return pitch_wave


def fluidsynth_without_normalize(self, fs=44100, sf2_path=None):
    """Synthesize using fluidsynth. without signal normalize
    Parameters
    ----------
    fs : int
        Sampling rate to synthesize at.
    sf2_path : str
        Path to a .sf2 file.
        Default ``None``, which uses the TimGM6mb.sf2 file included with
        ``pretty_midi``.
    Returns
    -------
    synthesized : np.ndarray
        Waveform of the MIDI data, synthesized at ``fs``.
    """
    # If there are no instruments, or all instruments have no notes, return
    # an empty array
    if len(self.instruments) == 0 or all(len(i.notes) == 0 for i in self.instruments):
        return np.array([])
    # Get synthesized waveform for each instrument
    waveforms = [i.fluidsynth(fs=fs, sf2_path=sf2_path) for i in self.instruments]
    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[: waveform.shape[0]] += waveform
    # Normalize
    # synthesized /= np.abs(synthesized).max()
    return synthesized
