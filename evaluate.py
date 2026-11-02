#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastSpeech 2 System Verification Suite
Verifies project structure, model checkpoints, audio waveforms, and variance plots.

Usage:
    python evaluate.py
    python evaluate.py --live-inference
"""

import os
import sys
import wave
import struct
import math
import argparse

# Enable UTF-8 standard output for Windows terminals
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def check_lfs_pointer(path):
    """Checks whether a file on disk is an unpulled Git LFS text pointer."""
    if not os.path.exists(path):
        return False, "File missing"
    size = os.path.getsize(path)
    if size < 1024:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(256)
                if "git-lfs" in content or "oid sha256" in content:
                    return True, f"Git LFS pointer text ({size} bytes)"
        except Exception:
            pass
    return False, f"Binary asset ({size / (1024*1024):.2f} MB)"


def inspect_wav_file(wav_path):
    """Inspects audio wav file properties without third-party dependencies."""
    if not os.path.exists(wav_path):
        return None, "File not found"
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / float(framerate)

            frames = wf.readframes(n_frames)
            fmt = f"<{n_frames * n_channels}h" if sampwidth == 2 else None
            if fmt and len(frames) == struct.calcsize(fmt):
                samples = struct.unpack(fmt, frames)
                norm_samples = [s / 32768.0 for s in samples]
                peak = max(abs(s) for s in norm_samples)
                rms = math.sqrt(sum(s * s for s in norm_samples) / max(1, len(norm_samples)))
            else:
                peak = 0.0
                rms = 0.0

            return {
                "channels": n_channels,
                "sampwidth": sampwidth,
                "framerate": framerate,
                "duration": duration,
                "peak": peak,
                "rms": rms,
            }, "OK"
    except Exception as e:
        return None, str(e)


def audit_structure():
    """Validates module and resource layout."""
    expected_structure = [
        "English/train.py",
        "English/preprocess.py",
        "English/inference.py",
        "English/config.yaml",
        "English/checkpoints",
        "Telugu/train.py",
        "Telugu/preprocess.py",
        "Telugu/inference.py",
        "Telugu/config.yaml",
        "Telugu/checkpoints",
        "Results/English/training_loss.png",
        "Results/English/validation_loss.png",
        "Results/English/test_generated_audios",
        "Results/English/variance_adaptor_plots/duration",
        "Results/English/variance_adaptor_plots/pitch",
        "Results/English/variance_adaptor_plots/energy",
        "Results/Telugu/training_loss.png",
        "Results/Telugu/validation_loss.png",
        "Results/Telugu/test_generated_audios",
        "Results/Telugu/variance_adaptor_plots/duration",
        "Results/Telugu/variance_adaptor_plots/pitch",
        "Results/Telugu/variance_adaptor_plots/energy",
        "Report/Technical_Report.pdf",
        "README.md",
        "requirements.txt",
    ]

    missing = []
    for item in expected_structure:
        full_path = os.path.join(ROOT_DIR, item)
        if not os.path.exists(full_path):
            missing.append(item)
    return missing


def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 System Verification Suite")
    parser.add_argument("--live-inference", action="store_true", help="Run live neural synthesis test on CPU/GPU")
    args = parser.parse_args()

    print("=" * 78)
    print("            FASTSPEECH 2 SYSTEM AND ASSET VERIFICATION")
    print("=" * 78)
    print(f"Repository Root : {ROOT_DIR}")
    print(f"Python Runtime  : {sys.version.split()[0]}")
    print("-" * 78)

    # 1. Directory Structure Audit
    print("\n[1/6] Verifying Project Layout and Modules...")
    missing = audit_structure()
    if missing:
        print(f"  [FAIL] Missing items ({len(missing)}):")
        for m in missing:
            print(f"         - {m}")
    else:
        print("  [PASS] All expected modules, scripts, and asset directories are present.")

    # 2. Checkpoint Audit
    print("\n[2/6] Verifying Acoustic and Vocoder Checkpoints...")
    ckpts = [
        ("English FastSpeech 2 (Best)", os.path.join(ROOT_DIR, "English", "checkpoints", "checkpoint_best.pth.tar")),
        ("English FastSpeech 2 (Final)", os.path.join(ROOT_DIR, "English", "checkpoints", "checkpoint_final.pth.tar")),
        ("Telugu FastSpeech 2 (Best)", os.path.join(ROOT_DIR, "Telugu", "checkpoints", "checkpoint_best.pth.tar")),
        ("Telugu FastSpeech 2 (Final)", os.path.join(ROOT_DIR, "Telugu", "checkpoints", "checkpoint_final.pth.tar")),
        ("Universal HiFi-GAN Vocoder", os.path.join(ROOT_DIR, "core", "hifigan", "generator_universal.pth.tar")),
    ]
    all_ckpts_valid = True
    for label, p in ckpts:
        is_lfs, info = check_lfs_pointer(p)
        if not os.path.exists(p):
            print(f"  [FAIL] {label}: Not found at {p}")
            all_ckpts_valid = False
        elif is_lfs:
            print(f"  [WARN] {label}: {info} (Run 'git lfs pull')")
            all_ckpts_valid = False
        else:
            print(f"  [PASS] {label}: {info}")

    # 3. Speech Waveforms Verification
    print("\n[3/6] Verifying Synthesized Speech Waveforms (10 Utterances)...")
    english_ids = ["E1", "E2a", "E2b", "E3", "E4"]
    telugu_ids = ["T1", "T2a", "T2b", "T3", "T4"]
    audio_valid = True

    print("  English Utterances (Results/English/test_generated_audios/):")
    for s_id in english_ids:
        wav_path = os.path.join(ROOT_DIR, "Results", "English", "test_generated_audios", f"{s_id}.wav")
        meta, status = inspect_wav_file(wav_path)
        if meta and meta["framerate"] == 22050 and meta["rms"] > 0.05:
            print(f"    [PASS] {s_id}.wav: {meta['framerate']} Hz, {meta['duration']:.2f}s, RMS={meta['rms']:.4f}, Peak={meta['peak']:.4f}")
        else:
            print(f"    [FAIL] {s_id}.wav: {status}")
            audio_valid = False

    print("  Telugu Utterances (Results/Telugu/test_generated_audios/):")
    for s_id in telugu_ids:
        wav_path = os.path.join(ROOT_DIR, "Results", "Telugu", "test_generated_audios", f"{s_id}.wav")
        meta, status = inspect_wav_file(wav_path)
        if meta and meta["framerate"] == 22050 and meta["rms"] > 0.05:
            print(f"    [PASS] {s_id}.wav: {meta['framerate']} Hz, {meta['duration']:.2f}s, RMS={meta['rms']:.4f}, Peak={meta['peak']:.4f}")
        else:
            print(f"    [FAIL] {s_id}.wav: {status}")
            audio_valid = False

    # 4. Variance Adaptor Plots Audit
    print("\n[4/6] Verifying Variance Adaptor Visualizations (30 Plots)...")
    plots_valid = True
    plot_types = ["duration", "pitch", "energy"]
    en_plot_count = 0
    te_plot_count = 0

    for s_id in english_ids:
        for p_type in plot_types:
            path = os.path.join(ROOT_DIR, "Results", "English", "variance_adaptor_plots", p_type, f"{s_id}_{p_type}.png")
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                en_plot_count += 1
            else:
                plots_valid = False

    for s_id in telugu_ids:
        for p_type in plot_types:
            path = os.path.join(ROOT_DIR, "Results", "Telugu", "variance_adaptor_plots", p_type, f"{s_id}_{p_type}.png")
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                te_plot_count += 1
            else:
                plots_valid = False

    print(f"  English Variance Plots : {en_plot_count}/15 verified")
    print(f"  Telugu Variance Plots  : {te_plot_count}/15 verified")
    if plots_valid and en_plot_count == 15 and te_plot_count == 15:
        print("  [PASS] All 30 duration, pitch (F0 in Hz), and energy contour plots verified.")
    else:
        print("  [FAIL] Incomplete variance adaptor plots.")

    # 5. Loss Plots and Technical Report Audit
    print("\n[5/6] Verifying Multi-Task Loss Curves & Technical Report...")
    loss_curves = [
        "Results/English/training_loss.png",
        "Results/English/validation_loss.png",
        "Results/Telugu/training_loss.png",
        "Results/Telugu/validation_loss.png",
    ]
    loss_valid = True
    for lc in loss_curves:
        p = os.path.join(ROOT_DIR, lc)
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            print(f"  [PASS] {lc} ({os.path.getsize(p)/1024:.1f} KB)")
        else:
            print(f"  [FAIL] {lc} missing or corrupt")
            loss_valid = False

    report_path = os.path.join(ROOT_DIR, "Report", "Technical_Report.pdf")
    if os.path.exists(report_path) and os.path.getsize(report_path) > 50000:
        report_size_kb = os.path.getsize(report_path) / 1024
        print(f"  [PASS] Report/Technical_Report.pdf verified ({report_size_kb:.1f} KB)")
    else:
        print(f"  [FAIL] Report/Technical_Report.pdf missing or invalid")

    # 6. Live Inference Option
    if args.live_inference:
        print("\n[6/6] Executing Live Inference Verification...")
        try:
            from core.synthesizer import FastSpeech2Synthesizer
            synth = FastSpeech2Synthesizer(os.path.join(ROOT_DIR, "English", "config.yaml"))
            test_dir = os.path.join(ROOT_DIR, "Results", "eval_live_test")
            os.makedirs(test_dir, exist_ok=True)
            synth.synthesize_sentence(
                sentence_id="eval_test",
                text="FastSpeech 2 synthesis verification completed.",
                output_audio_dir=test_dir,
                output_plot_dir=test_dir
            )
            print(f"  [PASS] Live synthesis executed successfully -> {test_dir}")
        except Exception as e:
            print(f"  [FAIL] Live inference error: {e}")
    else:
        print("\n[6/6] Live inference test skipped (use --live-inference to run synthesis check).")

    # Final Summary Verdict
    print("\n" + "=" * 78)
    print("                         SYSTEM VERIFICATION SUMMARY")
    print("=" * 78)
    overall_pass = (not missing) and all_ckpts_valid and audio_valid and plots_valid and loss_valid
    if overall_pass:
        print("  STATUS : ALL CHECKS PASSED")
        print("  DETAILS: Checkpoints, speech audio, variance contours, and documentation verified.")
    else:
        print("  STATUS : ISSUES DETECTED")
        print("  DETAILS: Please inspect the flagged items above.")
    print("=" * 78)


if __name__ == "__main__":
    main()


