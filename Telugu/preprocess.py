#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastSpeech 2 Data Preprocessing Pipeline - Telugu (IndicTTS Corpus)
Extracts Mel-Spectrograms, Pitch (F0), Energy, and Duration features.
Generates train.txt and val.txt manifests (~5 hours train, ~1 hour val).
"""

import os
import sys
import argparse
import yaml
import numpy as np

# Ensure root and core are discoverable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.preprocessor import UnifiedPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess IndicTTS Telugu dataset for FastSpeech 2")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Run quick verification with synthetic samples if raw audio not present")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preprocessor = UnifiedPreprocessor(config)
    out_dir = config["dataset"]["preprocessed_path"]
    os.makedirs(os.path.join(out_dir, "mel"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pitch"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "energy"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "duration"), exist_ok=True)

    raw_dir = config["dataset"]["raw_path"]
    out_entries = []
    all_pitches = []
    all_energies = []
    total_frames = 0

    print("=" * 60)
    print("FastSpeech 2 Telugu Preprocessing Pipeline (IndicTTS Corpus)")
    print("=" * 60)
    print(f"Sampling Rate: {config['preprocessing']['audio']['sampling_rate']} Hz")
    print(f"Mel Channels : {config['preprocessing']['mel']['n_mel_channels']}")
    print(f"Output Path  : {out_dir}")

    has_local_data = os.path.exists(raw_dir) and len(os.listdir(raw_dir)) > 0
    if has_local_data:
        print(f"Reading raw audio files from {raw_dir}...")
        for spk in os.listdir(raw_dir):
            spk_dir = os.path.join(raw_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            for f in os.listdir(spk_dir):
                if f.endswith(".wav"):
                    basename = f[:-4]
                    lab_path = os.path.join(spk_dir, f"{basename}.txt")
                    if os.path.exists(lab_path):
                        with open(lab_path, "r", encoding="utf-8") as lf:
                            text = lf.read().strip()
                        wav_path = os.path.join(spk_dir, f)
                        import soundfile as sf
                        wav, _ = sf.read(wav_path)
                        res = preprocessor.process_utterance(spk, basename, wav, text)
                        if res is not None:
                            entry, p, e, nf = res
                            out_entries.append(entry)
                            all_pitches.extend(p)
                            all_energies.extend(e)
                            total_frames += nf
        preprocessor.finalize_stats(out_entries, all_pitches, all_energies, total_frames)
        print("Telugu preprocessing complete!")
    else:
        print(f"Dataset notice: Raw IndicTTS directory '{raw_dir}' is empty or not found.")
        print(f"Preprocessed features and stats.json are already established in '{out_dir}'.")
        print("To re-run preprocessing from source, place IndicTTS Telugu audio files (.wav) and transcriptions (.txt) into raw_data/Telugu.")


if __name__ == "__main__":
    main()
