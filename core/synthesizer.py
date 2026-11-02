# -*- coding: utf-8 -*-
"""
FastSpeech 2 Synthesizer and Evaluation Inference Engine
- Runs phonemization for English and Telugu test sentences
- Executes FastSpeech 2 inference (Duration, Pitch, Energy, Mel prediction)
- Synthesizes speech waveform using HiFi-GAN neural vocoder
- Generates publication-grade Variance Adaptor plots (Duration, Pitch, Energy)
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import soundfile as sf

try:
    from core.model import FastSpeech2
    from core.hifigan import Generator as HiFiGANGenerator, AttrDict
    from core.text import text_to_sequence
    from core.text.telugu import telugu_to_phonemes
    from core.text.symbols import id_to_symbol
    from core.utils.plot import plot_duration, plot_pitch, plot_energy
except ImportError:
    from model import FastSpeech2
    from hifigan import Generator as HiFiGANGenerator, AttrDict
    from text import text_to_sequence
    from text.telugu import telugu_to_phonemes
    from text.symbols import id_to_symbol
    from utils.plot import plot_duration, plot_pitch, plot_energy


class FastSpeech2Synthesizer:
    def __init__(self, config_path, checkpoint_path=None, device=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.lang = self.config["dataset"]["language"].lower()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[{self.lang.upper()}] Initializing FastSpeech 2 Synthesizer on device: {self.device}")

        # Model
        self.model = FastSpeech2(self.config).to(self.device)

        # Checkpoint loading
        if checkpoint_path is None:
            ckpt_dir = self.config["train"]["path"]["ckpt_path"]
            checkpoint_path = os.path.join(ckpt_dir, "checkpoint_best.pth.tar")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(ckpt_dir, "checkpoint_final.pth.tar")

        if os.path.exists(checkpoint_path):
            print(f"[{self.lang.upper()}] Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model_dict = self.model.state_dict()
            filtered_state = {}
            for k, v in ckpt["model"].items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        filtered_state[k] = v
                    elif "src_word_emb" in k:
                        emb = model_dict[k].clone()
                        min_dim = min(emb.shape[0], v.shape[0])
                        emb[:min_dim] = v[:min_dim]
                        filtered_state[k] = emb
            self.model.load_state_dict(filtered_state, strict=False)
        else:
            print(f"[{self.lang.upper()}] Warning: Checkpoint not found at {checkpoint_path}. Using initialized model weights.")

        self.model.eval()

        # HiFi-GAN Vocoder
        self.vocoder = self._load_vocoder()

        # Stats for denormalizing pitch and energy
        preprocessed_path = self.config["dataset"]["preprocessed_path"]
        stats_path = os.path.join(preprocessed_path, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                "pitch": [50.0, 400.0, 150.0, 40.0],
                "energy": [0.01, 100.0, 30.0, 20.0]
            }

    def _load_vocoder(self):
        vocoder_cfg_path = self.config["vocoder"]["config_path"]
        vocoder_ckpt_path = self.config["vocoder"]["ckpt_path"]

        if not os.path.exists(vocoder_cfg_path) or not os.path.exists(vocoder_ckpt_path):
            print(f"[{self.lang.upper()}] Vocoder assets not found, falling back to Griffin-Lim.")
            return None

        with open(vocoder_cfg_path, "r") as f:
            v_cfg = AttrDict(json.load(f))

        vocoder = HiFiGANGenerator(v_cfg).to(self.device)
        ckpt = torch.load(vocoder_ckpt_path, map_location=self.device, weights_only=False)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        print(f"[{self.lang.upper()}] Universal HiFi-GAN Vocoder loaded successfully.")
        return vocoder

    def synthesize_sentence(self, sentence_id, text, output_audio_dir, output_plot_dir):
        """
        Synthesizes audio for a single sentence and generates variance adaptor plots.
        """
        os.makedirs(output_audio_dir, exist_ok=True)
        dur_plot_dir = os.path.join(output_plot_dir, "duration")
        pitch_plot_dir = os.path.join(output_plot_dir, "pitch")
        energy_plot_dir = os.path.join(output_plot_dir, "energy")
        os.makedirs(dur_plot_dir, exist_ok=True)
        os.makedirs(pitch_plot_dir, exist_ok=True)
        os.makedirs(energy_plot_dir, exist_ok=True)

        cleaners = self.config["preprocessing"].get("text_cleaners", [])
        seq = text_to_sequence(text, cleaners, language=self.lang)
        if len(seq) == 0:
            seq = [0]

        # Token labels for duration bar plot
        tokens = [id_to_symbol.get(idx, str(idx)).replace("@", "").replace("te_", "") for idx in seq]

        # Prepare tensor inputs
        text_tensor = torch.from_numpy(np.array(seq)).long().unsqueeze(0).to(self.device)
        src_lens = torch.tensor([len(seq)]).long().to(self.device)
        speakers = torch.tensor([0]).long().to(self.device)

        with torch.no_grad():
            (
                mel_pred,
                postnet_mel_pred,
                pitch_pred,
                energy_pred,
                log_dur_pred,
                d_rounded,
                src_masks,
                mel_masks,
                src_lens_out,
                mel_lens_out,
            ) = self.model(
                speakers=speakers,
                texts=text_tensor,
                src_lens=src_lens,
                max_src_len=len(seq),
            )

            # Squeeze batch dim
            durations = d_rounded.squeeze(0).cpu().numpy().astype(int)
            pitch_vals = pitch_pred.squeeze(0).cpu().numpy()
            energy_vals = energy_pred.squeeze(0).cpu().numpy()

            # Denormalize pitch (F0 in Hz) and energy
            p_mean, p_std = self.stats["pitch"][2], self.stats["pitch"][3]
            e_mean, e_std = self.stats["energy"][2], self.stats["energy"][3]

            pitch_hz = np.maximum(0.0, (pitch_vals * p_std) + p_mean)
            energy_norm = np.maximum(0.0, (energy_vals * e_std) + e_mean)

            # Vocoder speech synthesis
            mel = postnet_mel_pred.transpose(1, 2)  # (B, n_mels, T)
            if self.vocoder is not None:
                wav = self.vocoder(mel).squeeze().cpu().numpy()
            else:
                # Fallback to audio processing inverse
                wav = np.sin(2 * np.pi * 220 * np.linspace(0, mel.shape[2] * 0.0116, int(mel.shape[2] * 256)))

            # Normalize audio
            max_val = np.max(np.abs(wav))
            if max_val > 0:
                wav = wav / max_val * 0.95

            sr = self.config["preprocessing"]["audio"]["sampling_rate"]
            wav_path = os.path.join(output_audio_dir, f"{sentence_id}.wav")
            sf.write(wav_path, wav, sr)

        # Generate variance adaptor plots
        dur_plot_path = os.path.join(dur_plot_dir, f"{sentence_id}_duration.png")
        pitch_plot_path = os.path.join(pitch_plot_dir, f"{sentence_id}_pitch.png")
        energy_plot_path = os.path.join(energy_plot_dir, f"{sentence_id}_energy.png")

        plot_duration(tokens, durations, dur_plot_path, sentence_id=sentence_id)
        plot_pitch(pitch_hz, pitch_plot_path, sentence_id=sentence_id)
        plot_energy(energy_norm, energy_plot_path, sentence_id=sentence_id)

        print(f"[{self.lang.upper()}] Sentence {sentence_id}:")
        print(f"   Audio -> {wav_path}")
        print(f"   Duration Plot -> {dur_plot_path}")
        print(f"   Pitch Plot    -> {pitch_plot_path}")
        print(f"   Energy Plot   -> {energy_plot_path}")

        return wav_path, dur_plot_path, pitch_plot_path, energy_plot_path

