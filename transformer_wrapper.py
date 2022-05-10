import os
import random

import numpy as np
import librosa
import torch
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from transformers import T5Config, T5ForConditionalGeneration, Adafactor

from midi_tokenizer import MidiTokenizer, extrapolate_beat_times
from layer.input import LogMelSpectrogram, ConcatEmbeddingToMel
from preprocess.beat_quantizer import extract_rhythm, interpolate_beat_times
from utils.dsp import get_stereo


DEFAULT_COMPOSERS = {"various composer": 2052}


class TransformerWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = MidiTokenizer(config.tokenizer)
        self.t5config = T5Config.from_pretrained("t5-small")

        for k, v in config.t5.items():
            self.t5config.__setattr__(k, v)

        self.transformer = T5ForConditionalGeneration(self.t5config)
        self.use_mel = self.config.dataset.use_mel
        self.mel_is_conditioned = self.config.dataset.mel_is_conditioned
        self.composer_to_feature_token = config.composer_to_feature_token

        if self.use_mel and not self.mel_is_conditioned:
            self.composer_to_feature_token = DEFAULT_COMPOSERS

        if self.use_mel:
            self.spectrogram = LogMelSpectrogram()
            if self.mel_is_conditioned:
                n_dim = 512
                composer_n_vocab = len(self.composer_to_feature_token)
                embedding_offset = min(self.composer_to_feature_token.values())
                self.mel_conditioner = ConcatEmbeddingToMel(
                    embedding_offset=embedding_offset,
                    n_vocab=composer_n_vocab,
                    n_dim=n_dim,
                )
        else:
            self.spectrogram = None

        self.lr = config.training.lr
        self.save_hyperparameters(self.config)
        self.saved_gt_to_tensorboard = 0

    def forward(self, input_ids, labels):
        """
        Deprecated.
        """
        rt = self.transformer(input_ids=input_ids, labels=labels)
        return rt

    @torch.no_grad()
    def single_inference(
        self,
        feature_tokens=None,
        audio=None,
        beatstep=None,
        max_length=256,
        max_batch_size=64,
        n_bars=None,
        composer_value=None,
    ):
        """
        generate a long audio sequence

        feature_tokens or audio : shape (time, )

        beatstep : shape (time, )
        - input_ids가 해당하는 beatstep 값들
        (offset 빠짐, 즉 beatstep[0] == 0)
        - beatstep[-1] : input_ids가 끝나는 지점의 시간값
        (즉 beatstep[-1] == len(y)//sr)
        """

        assert feature_tokens is not None or audio is not None
        assert beatstep is not None

        if feature_tokens is not None:
            assert len(feature_tokens.shape) == 1

        if audio is not None:
            assert len(audio.shape) == 1

        config = self.config
        PAD = self.t5config.pad_token_id
        n_bars = config.dataset.n_bars if n_bars is None else n_bars

        if beatstep[0] > 0.01:
            print(
                "inference warning : beatstep[0] is not 0 ({beatstep[0]}). all beatstep will be shifted."
            )
            beatstep = beatstep - beatstep[0]

        if self.use_mel:
            input_ids = None
            inputs_embeds, ext_beatstep = self.prepare_inference_mel(
                audio,
                beatstep,
                n_bars=n_bars,
                padding_value=PAD,
                composer_value=composer_value,
            )
            batch_size = inputs_embeds.shape[0]
        else:
            input_ids, ext_beatstep = self.prepare_inference_token(
                feature_tokens,
                beatstep,
                n_bars=n_bars,
                padding_value=PAD,
                use_composer=True,
            )
            inputs_embeds = None
            batch_size = input_ids.shape[0]

        # Considering GPU capacity, some sequence would not be generated at once.
        relative_tokens = list()
        for i in range(0, batch_size, max_batch_size):
            start = i
            end = min(batch_size, i + max_batch_size)

            if input_ids is None:
                _input_ids = None
                _inputs_embeds = inputs_embeds[start:end]
            else:
                _input_ids = input_ids[start:end]
                _inputs_embeds = None

            _relative_tokens = self.transformer.generate(
                input_ids=_input_ids,
                inputs_embeds=_inputs_embeds,
                max_length=max_length,
            )
            _relative_tokens = _relative_tokens.cpu().numpy()
            relative_tokens.append(_relative_tokens)

        max_length = max([rt.shape[-1] for rt in relative_tokens])
        for i in range(len(relative_tokens)):
            relative_tokens[i] = np.pad(
                relative_tokens[i],
                [(0, 0), (0, max_length - relative_tokens[i].shape[-1])],
                constant_values=PAD,
            )
        relative_tokens = np.concatenate(relative_tokens)

        pm, notes = self.tokenizer.relative_batch_tokens_to_midi(
            relative_tokens,
            beatstep=ext_beatstep,
            bars_per_batch=n_bars,
            cutoff_time_idx=(n_bars + 1) * 4,
        )

        return relative_tokens, notes, pm

    def prepare_inference_mel(
        self, audio, beatstep, n_bars, padding_value, composer_value=None
    ):
        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        sample_rate = self.config.dataset.sample_rate
        ext_beatstep = extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        def split_audio(audio):
            # Split audio corresponding beat intervals.
            # Each audio's lengths are different.
            # Because each corresponding beat interval times are different.
            batch = []

            for i in range(0, n_target_step, n_steps):

                start_idx = i
                end_idx = min(i + n_steps, n_target_step)

                start_sample = int(ext_beatstep[start_idx] * sample_rate)
                end_sample = int(ext_beatstep[end_idx] * sample_rate)
                feature = audio[start_sample:end_sample]
                batch.append(feature)
            return batch

        def pad_and_stack_batch(batch):
            batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
            return batch

        batch = split_audio(audio)
        batch = pad_and_stack_batch(batch)

        inputs_embeds = self.spectrogram(batch).transpose(-1, -2)
        if self.mel_is_conditioned:
            composer_value = torch.tensor(composer_value).to(self.device)
            composer_value = composer_value.repeat(inputs_embeds.shape[0])
            inputs_embeds = self.mel_conditioner(inputs_embeds, composer_value)
        return inputs_embeds, ext_beatstep

    def on_validation_epoch_start(self) -> None:
        return super().on_validation_epoch_start()

    def validation_step(self, mini_batch, batch_idx):
        loss = self.common_transcription_step(mini_batch, phase="val")
        return loss

    def validation_epoch_end(self, outputs) -> None:
        def to_gpu(dict_batch: dict):
            for k, v in dict_batch.items():
                if not isinstance(dict_batch[k], list):  # not string(path)
                    dict_batch[k] = v.cuda()
            return dict_batch

        writer = self.logger.experiment

        val_dataset = self.trainer.val_dataloaders[0].dataset

        val_dataset.eval = True
        mini_batch = next(iter(self.trainer.val_dataloaders[0]))
        mini_batch = to_gpu(mini_batch)
        val_dataset.eval = False

        def validation_log(audio_batch):
            rendered_midis = []
            transcripted_pm = []

            log_num = min(audio_batch["audio"].shape[0], 16)
            for i in range(log_num):
                filename = audio_batch["filename"][i]
                idx_in_dataset = audio_batch["idx"][i].item()
                beat_offset_idx = audio_batch["start_idx"][i].item()
                beat_end_idx = audio_batch["end_idx"][i].item()
                composer_value = audio_batch["composer_value"][i].item()
                beatstep = (
                    val_dataset.beatsteps[idx_in_dataset][beat_offset_idx:beat_end_idx]
                    - val_dataset.beatsteps[idx_in_dataset][beat_offset_idx]
                )
                midi_path = f"{self.logger._experiment.dir}/transcript/{filename}.midi"
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)
                feature_tokens = (
                    None if self.use_mel else audio_batch["feature_tokens"][i]
                )
                audio = audio_batch["audio"][i] if self.use_mel else None
                relative_tokens, notes, pm = self.single_inference(
                    feature_tokens=feature_tokens,
                    audio=audio,
                    beatstep=beatstep,
                    max_length=self.config.dataset.target_length,
                    composer_value=composer_value,
                )
                pm.write(midi_path)
                rendered_midi = pm.fluidsynth(self.config.dataset.sample_rate)
                transcripted_pm.append(pm)
                rendered_midis.append(rendered_midi)

            if self.saved_gt_to_tensorboard < 1:
                audio_logs = []
                for i in range(log_num):
                    filename = audio_batch["filename"][i]
                    audio_log = wandb.Audio(
                        audio_batch["audio"][i].cpu().numpy(),
                        sample_rate=self.config.dataset.sample_rate,
                        caption=f"{filename}",
                    )
                    audio_logs.append(audio_log)

                writer.log({"pop_audio": audio_logs})
                self.saved_gt_to_tensorboard += 1

            piano_logs = []
            for i in range(log_num):
                filename = audio_batch["filename"][i]
                if rendered_midis[i] is None or len(rendered_midis[i]) <= 0:
                    rendered_midis[i] = (
                        torch.zeros_like(audio_batch["audio"][i]).detach().cpu().numpy()
                    )

                audio_log = wandb.Audio(
                    rendered_midis[i],
                    sample_rate=self.config.dataset.sample_rate,
                    caption=f"{filename}",
                )
                piano_logs.append(audio_log)

            writer.log({"piano": piano_logs})

        if self.global_rank == 0:
            validation_log(mini_batch)

    def configure_optimizers(self):
        """
        optimizer, scheduler 세팅
        """
        config = self.config.training
        if config.optimizer == "adafactor":
            Optimizer = Adafactor
            kwargs = dict(relative_step=False, clip_threshold=config.gradient_clip_val)
        else:
            raise NotImplementedError

        transcriber_optimizer = Optimizer(
            self.transformer.parameters(), lr=self.lr, **kwargs
        )

        # Setting Scheduler
        monitor = None
        if config.lr_scheduler == "multistep":
            t_scheduler = optim.lr_scheduler.MultiStepLR(
                transcriber_optimizer,
                [15 * (x + 2) for x in range(500)],
                gamma=config.lr_decay,
            )

        elif not config.lr_scheduler:
            t_scheduler = None
        else:
            raise ValueError(f"unknown lr_scheduler :: {config.lr_scheduler}")

        if t_scheduler is None:
            return transcriber_optimizer
        else:
            return (
                [transcriber_optimizer],
                [t_scheduler],
            )

    @torch.no_grad()
    def generate(
        self,
        audio_path=None,
        composer=None,
        model="generated",
        steps_per_beat=2,
        stereo_amp=0.5,
        n_bars=2,
        ignore_duplicate=True,
        show_plot=False,
        save_midi=False,
        save_mix=False,
        midi_path=None,
        mix_path=None,
        click_amp=0.2,
        add_click=False,
        max_batch_size=None,
        vqvae=None,
        vqvae_token=None,
        beatsteps=None,
        mix_sample_rate=None,
        audio_y=None,
        audio_sr=None,
    ):
        if not self.use_mel:
            assert vqvae is not None or vqvae_token is not None

        if not audio_path and audio_y is None:
            assert not self.use_mel
            assert beatsteps is not None and vqvae_token is not None

        config = self.config
        device = self.device

        if audio_path is not None:
            extension = os.path.splitext(audio_path)[1]
            mix_path = (
                audio_path.replace(extension, f".{model}.{composer}.wav")
                if mix_path is None
                else mix_path
            )
            midi_path = (
                audio_path.replace(extension, f".{model}.{composer}.mid")
                if midi_path is None
                else midi_path
            )

        max_batch_size = 64 // n_bars if max_batch_size is None else max_batch_size
        composer_to_feature_token = self.composer_to_feature_token

        if composer is None:
            composer = random.sample(list(composer_to_feature_token.keys()), 1)[0]

        composer_value = composer_to_feature_token[composer]
        mix_sample_rate = (
            config.dataset.sample_rate if mix_sample_rate is None else mix_sample_rate
        )

        if not ignore_duplicate:
            if os.path.exists(midi_path):
                return

        ESSENTIA_SAMPLERATE = 44100

        if beatsteps is None:
            y, sr = librosa.load(audio_path, sr=ESSENTIA_SAMPLERATE)
            (
                bpm,
                beat_times,
                confidence,
                estimates,
                essentia_beat_intervals,
            ) = extract_rhythm(audio_path, y=y)
            beat_times = np.array(beat_times)
            beatsteps = interpolate_beat_times(beat_times, steps_per_beat, extend=True)
        else:
            y = None

        if self.use_mel:
            if audio_y is None and config.dataset.sample_rate != ESSENTIA_SAMPLERATE:
                if y is not None:
                    y = librosa.core.resample(
                        y,
                        orig_sr=ESSENTIA_SAMPLERATE,
                        target_sr=config.dataset.sample_rate,
                    )
                    sr = config.dataset.sample_rate
                else:
                    y, sr = librosa.load(audio_path, sr=config.dataset.sample_rate)
            elif audio_y is not None:
                if audio_sr != config.dataset.sample_rate:
                    audio_y = librosa.core.resample(
                        audio_y, orig_sr=audio_sr, target_sr=config.dataset.sample_rate
                    )
                    audio_sr = config.dataset.sample_rate
                y = audio_y
                sr = audio_sr

            start_sample = int(beatsteps[0] * sr)
            end_sample = int(beatsteps[-1] * sr)
            _audio = torch.from_numpy(y)[start_sample:end_sample].cuda()
            fzs = None
        else:
            raise NotImplementedError

        relative_tokens, notes, pm = self.single_inference(
            feature_tokens=fzs,
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            max_length=config.dataset.target_length
            * max(1, (n_bars // config.dataset.n_bars)),
            max_batch_size=max_batch_size,
            n_bars=n_bars,
            composer_value=composer_value,
        )

        for n in pm.instruments[0].notes:
            n.start += beatsteps[0]
            n.end += beatsteps[0]

        if show_plot or save_mix:
            if mix_sample_rate != sr:
                y = librosa.core.resample(y, orig_sr=sr, target_sr=mix_sample_rate)
                sr = mix_sample_rate
            if add_click:
                clicks = (
                    librosa.clicks(times=beatsteps, sr=sr, length=len(y)) * click_amp
                )
                y = y + clicks
            pm_y = pm.fluidsynth(sr)
            stereo = get_stereo(y, pm_y, pop_scale=stereo_amp)

        if show_plot:
            import IPython.display as ipd
            from IPython.display import display
            import note_seq

            display(ipd.Audio(pm_y, rate=sr))
            display(ipd.Audio(y, rate=sr))
            display(ipd.Audio(stereo, rate=sr))
            note_seq.plot_sequence(note_seq.midi_to_note_sequence(pm))

        if save_mix:
            sf.write(
                file=mix_path,
                data=stereo.T,
                samplerate=sr,
                format="wav",
            )

        if save_midi:
            pm.write(midi_path)

        return pm, composer, mix_path, midi_path
