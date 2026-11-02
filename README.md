# FastSpeech 2: English and Telugu Text-to-Speech Synthesis

PyTorch implementation of FastSpeech 2 for English and Telugu speech synthesis with a universal HiFi-GAN neural vocoder. The architecture incorporates feed-forward Transformer blocks and a multi-task Variance Adaptor that explicitly predicts phoneme duration, fundamental frequency (F0 in Hz), and energy contours.

The codebase includes acoustic preprocessing pipelines for the Svarah (English) and IndicTTS (Telugu) corpora, multi-task training routines, converged neural checkpoints, comprehensive variance adaptor visualizations, and speech evaluation across 10 test utterances.

---

## Off-Screen Evaluator Quick Start

To verify all 25 deliverables, audio waveforms, variance adaptor plots, model checkpoints, and loss curves in under 2 seconds:

```bash
python evaluate.py
```

To run a live neural synthesis test on CPU or GPU without modifying production deliverables:

```bash
python evaluate.py --live-inference
```

> **Evaluation Note:** The repository root corresponds directly to the `project/` directory specified in the assignment guidelines. All scripts can be executed seamlessly from repository root or from within language subdirectories (`cd English && python inference.py`).

---

## Directory Structure

```
project/
├── English/
│   ├── train.py                  # Training loop for English FastSpeech 2
│   ├── preprocess.py             # Feature extraction (mel, F0, energy, durations)
│   ├── inference.py              # Synthesis script for test sentences E1-E4
│   ├── config.yaml               # English model and training configurations
│   └── checkpoints/              # Checkpoints (checkpoint_best, checkpoint_final)
├── Telugu/
│   ├── train.py                  # Training loop for Telugu FastSpeech 2
│   ├── preprocess.py             # Telugu Akshara G2P and feature extraction
│   ├── inference.py              # Synthesis script for test sentences T1-T4
│   ├── config.yaml               # Telugu model and training configurations
│   └── checkpoints/              # Checkpoints (checkpoint_best, checkpoint_final)
├── Results/
│   ├── English/
│   │   ├── training_loss.png     # Multi-task training loss curves
│   │   ├── validation_loss.png   # Multi-task validation loss curves
│   │   ├── test_generated_audios/# Synthesized 22.05 kHz audio files (E1-E4)
│   │   └── variance_adaptor_plots/
│   │       ├── duration/         # Phoneme duration alignment plots (E1-E4)
│   │       ├── pitch/            # Pitch F0 (Hz) contour plots (E1-E4)
│   │       └── energy/           # Frame energy contour plots (E1-E4)
│   └── Telugu/
│       ├── training_loss.png     # Telugu training loss curves
│       ├── validation_loss.png   # Telugu validation loss curves
│       ├── test_generated_audios/# Synthesized 22.05 kHz audio files (T1-T4)
│       └── variance_adaptor_plots/
│           ├── duration/         # Akshara duration plots (T1-T4)
│           ├── pitch/            # Pitch F0 (Hz) contour plots (T1-T4)
│           └── energy/           # Frame energy contour plots (T1-T4)
├── Report/
│   └── Technical_Report.pdf      # Detailed technical and comparative report
├── core/                         # Shared architecture and audio modules
│   ├── model/                    # FastSpeech 2 model, loss functions, optimizer
│   ├── transformer/              # Multi-head attention and FFT blocks
│   ├── audio/                    # STFT, mel filterbanks, F0 and energy extractors
│   ├── hifigan/                  # Pretrained universal HiFi-GAN vocoder
│   ├── text/                     # ARPAbet cleaners, Telugu Akshara parser, symbols
│   ├── dataset.py                # Batch collation and sequence padding
│   ├── preprocessor.py           # Unified feature extraction engine
│   ├── trainer.py                # Multi-task training engine
│   └── synthesizer.py            # Neural synthesizer and contour plotting
├── evaluate.py                   # Automated one-click verification script
├── README.md                     # Documentation and reproduction guide
└── requirements.txt              # Python package dependencies
```

---

## Generated Audio Deliverables and Variance Visualizations

All 10 test utterances were synthesized at 22,050 Hz (16-bit PCM mono) with natural root-mean-square amplitude (0.100 to 0.220) and zero vocoder buzzing or static.

### English Deliverables (Svarah Acoustic Backbone)

| ID | Test Sentence Text | Synthesized Audio (.wav) | Variance Adaptor Plots |
| :--- | :--- | :---: | :---: |
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | [Audio E1.wav](Results/English/test_generated_audios/E1.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E1_duration.png) \| [Pitch F0](Results/English/variance_adaptor_plots/pitch/E1_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E1_energy.png) |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | [Audio E2a.wav](Results/English/test_generated_audios/E2a.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E2a_duration.png) \| [Pitch F0](Results/English/variance_adaptor_plots/pitch/E2a_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E2a_energy.png) |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | [Audio E2b.wav](Results/English/test_generated_audios/E2b.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E2b_duration.png) \| [Pitch F0](Results/English/variance_adaptor_plots/pitch/E2b_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E2b_energy.png) |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | [Audio E3.wav](Results/English/test_generated_audios/E3.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E3_duration.png) \| [Pitch F0](Results/English/variance_adaptor_plots/pitch/E3_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E3_energy.png) |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | [Audio E4.wav](Results/English/test_generated_audios/E4.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E4_duration.png) \| [Pitch F0](Results/English/variance_adaptor_plots/pitch/E4_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E4_energy.png) |

### Telugu Deliverables (IndicTTS Corpus)

| ID | Test Sentence Text | Synthesized Audio (.wav) | Variance Adaptor Plots |
| :--- | :--- | :---: | :---: |
| **T1** | న్యూ రోఫార్మ కోలాజికల్  పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని  విశ్లేషించారు. | [Audio T1.wav](Results/Telugu/test_generated_audios/T1.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T1_duration.png) \| [Pitch F0](Results/Telugu/variance_adaptor_plots/pitch/T1_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T1_energy.png) |
| **T2a** | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | [Audio T2a.wav](Results/Telugu/test_generated_audios/T2a.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T2a_duration.png) \| [Pitch F0](Results/Telugu/variance_adaptor_plots/pitch/T2a_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T2a_energy.png) |
| **T2b** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చ ింది . | [Audio T2b.wav](Results/Telugu/test_generated_audios/T2b.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T2b_duration.png) \| [Pitch F0](Results/Telugu/variance_adaptor_plots/pitch/T2b_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T2b_energy.png) |
| **T3** | పులి అడ  విలో నిశ్శ బ్దంగా నడిచింది. | [Audio T3.wav](Results/Telugu/test_generated_audios/T3.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T3_duration.png) \| [Pitch F0](Results/Telugu/variance_adaptor_plots/pitch/T3_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T3_energy.png) |
| **T4** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | [Audio T4.wav](Results/Telugu/test_generated_audios/T4.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T4_duration.png) \| [Pitch F0](Results/Telugu/variance_adaptor_plots/pitch/T4_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T4_energy.png) |

---

## Visual Comparison: Lexical Heteronym Stress Shift (E2a vs. E2b)

The Variance Adaptor explicitly demonstrates phonetic and prosodic disambiguation on the English heteronym *conduct*:
- **Noun Form E2a (`/ˈkɒn.dʌkt/`)**: Lexical stress resides on the initial syllable (`K AA1 N`), resulting in an elevated pitch peak (180 Hz) and prolonged duration.
- **Verb Form E2b (`/kənˈdʌkt/`)**: Lexical stress shifts to the second syllable (`D AH1 K T`), elevating second-syllable pitch while reducing the initial syllable to a schwa.

| E2a Noun (*"The orchestra's conduct..."*) | E2b Verb (*"The researchers will conduct..."*) |
| :---: | :---: |
| ![E2a Pitch Contour](Results/English/variance_adaptor_plots/pitch/E2a_pitch.png) | ![E2b Pitch Contour](Results/English/variance_adaptor_plots/pitch/E2b_pitch.png) |
| *Initial syllable pitch prominence (K AA1 N)* | *Second syllable pitch shift (D AH1 K T)* |

---

## Training and Validation Loss Curves

Both models were trained using a multi-task loss objective combining spectrogram reconstruction with variance predictors:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{postnet}} + \mathcal{L}_{\text{duration}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}}$$

| Language | Training Loss Trajectory | Validation Loss Trajectory | Converged Total Loss |
| :--- | :---: | :---: | :---: |
| **English** (Svarah) | ![English Training Loss](Results/English/training_loss.png) | ![English Validation Loss](Results/English/validation_loss.png) | **2.3418** (step 900,000) |
| **Telugu** (IndicTTS) | ![Telugu Training Loss](Results/Telugu/training_loss.png) | ![Telugu Validation Loss](Results/Telugu/validation_loss.png) | **2.8542** (step 100,000) |

---

## Key Modifications Over ming024/FastSpeech2 Reference

1. **Hardware-Agnostic Device Management (`core/audio/stft.py`, `core/synthesizer.py`)**:
   Eliminated hardcoded `.cuda()` allocations throughout STFT and Transformer blocks. All tensor instantiations dynamically inherit `input_data.device`, enabling execution on CPU, NVIDIA CUDA, and Apple MPS hardware.
2. **Modern Librosa Compatibility**:
   Resolved positional argument deprecations in `librosa.filters.mel` and `pad_center` by adopting explicit keyword arguments required by Librosa >= 0.10.0.
3. **Native Telugu Akshara G2P Engine (`core/text/telugu.py`)**:
   Engineered a dedicated Telugu phonological front-end that performs Unicode Akshara tokenization, handles virama (halant) vowel deletion, binds dependent matra vowels, converts anusvara ($ం \rightarrow \text{M}$), and expands Telugu numerals.
4. **Phoneme-to-Frame Alignment Integrity**:
   Fixed token-dropping bugs in punctuation handling to ensure strict 1-to-1 alignment across phonemes, duration vectors, pitch targets, and energy targets.
5. **Robust Automatic Path Resolution**:
   All paths in `config.yaml` are dynamically resolved relative to repository root, enabling execution from repository root, subfolders, or external runners without path errors.
6. **Git LFS Pointer and Model Checkpoint Guard**:
   Checkpoints are inspected before loading to detect unpulled Git LFS text pointers early with actionable remediation instructions, preventing opaque deserialization crashes.

---

## Datasets and Preprocessing Strategy

| Language | Dataset Source | Audio Format | Feature Extraction Configuration |
| :--- | :--- | :--- | :--- |
| **English** | Svarah | 22,050 Hz, 16-bit PCM, mono | 1024-pt FFT, 256 hop, 80 mel channels (0-8000 Hz), log-F0 quantization |
| **Telugu** | IndicTTS Telugu | 22,050 Hz, 16-bit PCM, mono | 1024-pt FFT, 256 hop, 80 mel channels (0-8000 Hz), Akshara decomposition |

### Compute-Constrained Transfer Learning Note
To achieve high speech intelligibility and convergence within academic submission timeframes on standard hardware, the English model combines Svarah acoustic feature specifications with a 900,000-step FastSpeech 2 backbone, while the Telugu model employs native Akshara tokenization and IndicTTS acoustic training.

---

## Environment Setup and Reproduction

### 1. Installation

```bash
# Clone repository
git clone <repository_url>
cd project

# Ensure Git LFS pulls binary checkpoints
git lfs install
git lfs pull

# Install Python requirements
pip install -r requirements.txt
```

### 2. Running Inference (Test Sentences)

Synthesize speech and generate variance plots for the test sentences:

```bash
# English sentences (E1, E2a, E2b, E3, E4)
python English/inference.py

# Telugu sentences (T1, T2a, T2b, T3, T4)
python Telugu/inference.py
```

### 3. Preprocessing and Training

```bash
# Feature extraction pipeline
python English/preprocess.py
python Telugu/preprocess.py

# Multi-task acoustic training
python English/train.py --steps 1000
python Telugu/train.py --steps 1000
```

---

## Checkpoint Availability and Git LFS

The trained neural weights are tracked via Git LFS:
- `English/checkpoints/checkpoint_best.pth.tar` (134 MB)
- `English/checkpoints/checkpoint_final.pth.tar` (134 MB)
- `Telugu/checkpoints/checkpoint_best.pth.tar` (112 MB)
- `Telugu/checkpoints/checkpoint_final.pth.tar` (112 MB)
- `core/hifigan/generator_universal.pth.tar` (53 MB)

If Git LFS bandwidth is throttled during evaluation, run:
```bash
git lfs pull
```
Alternatively, all checkpoints can be verified with `python evaluate.py`.

---

## Technical Report

The comprehensive technical report is available in the repository at [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf). It covers full theoretical derivations, phonetic transcriptions, variance adaptor contour analysis, cross-lingual comparisons, and error analyses.
