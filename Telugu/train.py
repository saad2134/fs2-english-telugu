#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastSpeech 2 Training Script - Telugu Model (IndicTTS Corpus)
Executes training loop, computes loss curves, and saves checkpoints in Telugu/checkpoints/.
"""

import os
import sys
import argparse

# Ensure root and core are discoverable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.trainer import FastSpeech2Trainer


def main():
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 Telugu Model")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "config.yaml"), help="Path to Telugu config.yaml")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--eval_interval", type=int, default=25, help="Interval for validation evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    trainer = FastSpeech2Trainer(args.config, device=args.device)
    trainer.train(target_steps=args.steps, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()

