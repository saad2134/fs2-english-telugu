# -*- coding: utf-8 -*-
"""
FastSpeech 2 PyTorch Dataset & DataLoader
Supports unified configs, dynamic language routing, and variable length batching.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

try:
    from core.text import text_to_sequence
    from core.utils.tools import pad_1D, pad_2D
except ImportError:
    from text import text_to_sequence
    from utils.tools import pad_1D, pad_2D


class FastSpeech2Dataset(Dataset):
    def __init__(self, filename, config, sort=False, drop_last=False):
        self.config = config
        self.preprocessed_path = config["dataset"]["preprocessed_path"]
        self.cleaners = config["preprocessing"].get("text_cleaners", [])
        self.lang = config["dataset"]["language"].lower()
        self.batch_size = config["train"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)
        
        spk_path = os.path.join(self.preprocessed_path, "speakers.json")
        if os.path.exists(spk_path):
            with open(spk_path) as f:
                self.speaker_map = json.load(f)
        else:
            self.speaker_map = {self.lang: 0}
            
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map.get(speaker, 0)
        raw_text = self.raw_text[idx]
        
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners, language=self.lang))

        mel_path = os.path.join(self.preprocessed_path, "mel", f"{speaker}-mel-{basename}.npy")
        mel = np.load(mel_path)

        pitch_path = os.path.join(self.preprocessed_path, "pitch", f"{speaker}-pitch-{basename}.npy")
        pitch = np.load(pitch_path)

        energy_path = os.path.join(self.preprocessed_path, "energy", f"{speaker}-energy-{basename}.npy")
        energy = np.load(energy_path)

        duration_path = os.path.join(self.preprocessed_path, "duration", f"{speaker}-duration-{basename}.npy")
        duration = np.load(duration_path)

        # Ensure exact dimension alignment between phoneme sequence and target features
        min_l = min(len(phone), len(pitch), len(energy), len(duration))
        phone = phone[:min_l]
        pitch = pitch[:min_l]
        energy = energy[:min_l]
        duration = duration[:min_l]

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        return sample

    def process_meta(self, filename):
        full_path = os.path.join(self.preprocessed_path, filename) if not os.path.isabs(filename) and not os.path.exists(filename) else filename
        name = []
        speaker = []
        text = []
        raw_text = []
        if not os.path.exists(full_path):
            return name, speaker, text, raw_text

        with open(full_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    n, s, t, r = parts[0], parts[1], parts[2], parts[3]
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
        return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens) if len(text_lens) > 0 else 0,
            mels,
            mel_lens,
            max(mel_lens) if len(mel_lens) > 0 else 0,
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist() if len(idx_arr) > 0 else []
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
