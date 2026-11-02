# -*- coding: utf-8 -*-
"""
Visualization utilities for FastSpeech 2:
- Training and validation loss curves (total, mel, postnet, duration, pitch, energy)
- Publication-quality Variance Adaptor plots (Duration, Pitch F0, and Energy contours)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(steps, losses, save_path, title="Training Loss Curve"):
    """
    Plots comprehensive loss decomposition curves:
    Total Loss, Mel Loss, PostNet Mel Loss, Pitch Loss, Energy Loss, Duration Loss.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    colors = {
        "Total": "#1f77b4",
        "Mel": "#ff7f0e",
        "PostNet": "#2ca02c",
        "Pitch": "#d62728",
        "Energy": "#9467bd",
        "Duration": "#8c564b"
    }
    
    keys = ["Total", "Mel", "PostNet", "Pitch", "Energy", "Duration"]
    for idx, (ax, k) in enumerate(zip(axes.flatten(), keys)):
        if k in losses and len(losses[k]) > 0:
            ax.plot(steps, losses[k], label=f"{k} Loss", color=colors[k], linewidth=2.0)
            ax.set_title(f"{k} Loss", fontsize=13, fontweight="semibold")
            ax.set_xlabel("Steps", fontsize=11)
            ax.set_ylabel("Loss Value", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", framealpha=0.9)
        else:
            ax.set_visible(False)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_duration(tokens, durations, save_path, sentence_id=""):
    """
    Plots predicted phoneme duration bar chart with phoneme token alignment.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    plt.figure(figsize=(max(10, len(tokens) * 0.35), 4.5), dpi=300)
    indices = np.arange(len(tokens))
    bars = plt.bar(indices, durations, color="#2b5c8f", edgecolor="#1a365d", width=0.7, alpha=0.85)
    
    plt.title(f"Predicted Duration Distribution - Sentence {sentence_id}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Phoneme Tokens", fontsize=11)
    plt.ylabel("Duration (Mel Frames)", fontsize=11)
    plt.xticks(indices, tokens, rotation=60, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.2, f"{int(h)}", ha="center", va="bottom", fontsize=7.5)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pitch(pitch_contour, save_path, sentence_id="", sr_hop_ms=11.6):
    """
    Plots predicted Pitch (F0) contour across time frames.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    pitch_contour = np.array(pitch_contour).squeeze()
    time_axis = np.arange(len(pitch_contour)) * (sr_hop_ms / 1000.0)
    
    plt.figure(figsize=(10, 4.5), dpi=300)
    plt.plot(time_axis, pitch_contour, color="#d9534f", linewidth=2.2, label="Predicted F0 Contour")
    plt.fill_between(time_axis, pitch_contour, color="#d9534f", alpha=0.15)
    
    plt.title(f"Predicted Pitch (F0) Trajectory - Sentence {sentence_id}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Fundamental Frequency F0 (Hz / Norm)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_energy(energy_contour, save_path, sentence_id="", sr_hop_ms=11.6):
    """
    Plots predicted Energy contour across time frames.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    energy_contour = np.array(energy_contour).squeeze()
    time_axis = np.arange(len(energy_contour)) * (sr_hop_ms / 1000.0)
    
    plt.figure(figsize=(10, 4.5), dpi=300)
    plt.plot(time_axis, energy_contour, color="#2e7d32", linewidth=2.2, label="Predicted Energy Contour")
    plt.fill_between(time_axis, energy_contour, color="#2e7d32", alpha=0.15)
    
    plt.title(f"Predicted Energy Trajectory - Sentence {sentence_id}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Frame Energy (L2 Norm / Norm)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

