#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastSpeech 2 Inference Script - Telugu Model (IndicTTS Corpus)
Generates audio speech samples and variance adaptor plots for all specified test sentences:
- T1: Phonetic loan adaptations and agglutinative conjuncts
- T2a: Noun polysemy ('పట్టు' as grip/hold)
- T2b: Adjectival polysemy ('పట్టు' as silk) and spacing nuance ('వచ్చ ింది')
- T3: Intra-word spacing artifact ('అడ విలో') and irregular anusvara
- T4: Temporal/numerical expressions (2026, 12వ, 10:45) and spelling variation ('సిద్ంధ గా')
"""

import os
import sys
import argparse

# Enable utf-8 standard output for Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.synthesizer import FastSpeech2Synthesizer

TELUGU_TEST_SAMPLES = [
    {
        "id": "T1",
        "text": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు."
    },
    {
        "id": "T2a",
        "text": "తాడుపై అతని పట్టుచాలా బలంగా ఉంది."
    },
    {
        "id": "T2b",
        "text": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది."
    },
    {
        "id": "T3",
        "text": "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది."
    },
    {
        "id": "T4",
        "text": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి."
    }
]


def main():
    parser = argparse.ArgumentParser(description="Run FastSpeech 2 Telugu Inference")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "config.yaml"), help="Path to Telugu config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth.tar)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--output_audio_dir", type=str, default=os.path.join(ROOT_DIR, "Results", "Telugu", "test_generated_audios"))
    parser.add_argument("--output_plot_dir", type=str, default=os.path.join(ROOT_DIR, "Results", "Telugu", "variance_adaptor_plots"))
    args = parser.parse_args()

    synthesizer = FastSpeech2Synthesizer(args.config, checkpoint_path=args.checkpoint, device=args.device)

    print("=" * 60)
    print("Running FastSpeech 2 Inference for Telugu Test Sentences")
    print("=" * 60)

    for sample in TELUGU_TEST_SAMPLES:
        s_id = sample["id"]
        text = sample["text"]
        print(f"\nProcessing [{s_id}]: \"{text}\"")
        synthesizer.synthesize_sentence(
            sentence_id=s_id,
            text=text,
            output_audio_dir=args.output_audio_dir,
            output_plot_dir=args.output_plot_dir
        )

    print("\nTelugu synthesis completed successfully!")


if __name__ == "__main__":
    main()
