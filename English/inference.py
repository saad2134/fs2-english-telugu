#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastSpeech 2 Inference Script - English Model (Svarah Corpus)
Generates audio speech samples and variance adaptor plots for all specified test sentences:
- E1: Polysyllabic morphology ('antidisestablishmentarianism')
- E2a: Noun heteronym prosody ('conduct' in orchestra context)
- E2b: Verb heteronym prosody ('conduct' in experiment context)
- E3: Typographic token split anomaly ('rhyth m')
- E4: Numerical, temporal, and code-switched lexical item ('andaroo')
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.synthesizer import FastSpeech2Synthesizer

ENGLISH_TEST_SAMPLES = [
    {
        "id": "E1",
        "text": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."
    },
    {
        "id": "E2a",
        "text": "The orchestra's conduct during the ceremony was praised by the audience."
    },
    {
        "id": "E2b",
        "text": "The researchers will conduct an experiment to evaluate the proposed method."
    },
    {
        "id": "E3",
        "text": "The rhyth m of the orchestra changed unexpectedly."
    },
    {
        "id": "E4",
        "text": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    }
]


def main():
    parser = argparse.ArgumentParser(description="Run FastSpeech 2 English Inference")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "config.yaml"), help="Path to English config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth.tar)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--output_audio_dir", type=str, default=os.path.join(ROOT_DIR, "Results", "English", "test_generated_audios"))
    parser.add_argument("--output_plot_dir", type=str, default=os.path.join(ROOT_DIR, "Results", "English", "variance_adaptor_plots"))
    args = parser.parse_args()

    synthesizer = FastSpeech2Synthesizer(args.config, checkpoint_path=args.checkpoint, device=args.device)

    print("=" * 60)
    print("Running FastSpeech 2 Inference for English Test Sentences")
    print("=" * 60)

    for sample in ENGLISH_TEST_SAMPLES:
        s_id = sample["id"]
        text = sample["text"]
        print(f"\nProcessing [{s_id}]: \"{text}\"")
        synthesizer.synthesize_sentence(
            sentence_id=s_id,
            text=text,
            output_audio_dir=args.output_audio_dir,
            output_plot_dir=args.output_plot_dir
        )

    print("\nEnglish synthesis completed successfully!")


if __name__ == "__main__":
    main()

