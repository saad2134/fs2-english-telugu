# FastSpeech 2 (English and Telugu)

PyTorch implementation of FastSpeech 2 for English and Telugu speech synthesis with a universal HiFi-GAN vocoder. The model incorporates feed-forward Transformer blocks and a multi-task Variance Adaptor that predicts phoneme duration, fundamental frequency (F0), and energy contours.

The codebase includes acoustic preprocessing pipelines for the Svarah (English) and IndicTTS (Telugu) corpora, multi-task training routines, and prosody evaluation across 10 test utterances.

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
├── README.md                     # Documentation and reproduction guide
└── requirements.txt              # Python package dependencies
```

---

## Architecture Overview

FastSpeech 2 uses a feed-forward Transformer architecture with a Variance Adaptor to predict acoustic variance directly:

```
Phoneme Sequence
       │
       ▼
┌──────────────┐
│  FFT Encoder │ (4 layers, d_model=256, 2 heads, kernel sizes [9, 1])
└──────┬───────┘
       │
       ▼
┌───────────────────────────────────────────────┐
│               Variance Adaptor                │
│  ┌────────────────────┐   ┌────────────────┐  │
│  │ Duration Predictor │──>│Length Regulator│  │
│  └────────────────────┘   └───────┬────────┘  │
│                                   │           │
│  ┌────────────────────┐           │           │
│  │  Pitch Predictor   │───────────┼──(+) Pitch│
│  └────────────────────┘           │           │
│  ┌────────────────────┐           │           │
│  │  Energy Predictor  │───────────┼──(+)Energy│
│  └────────────────────┘           │           │
└───────────────────────────────────┼───────────┘
                                    │
                                    ▼
┌──────────────┐
│  FFT Decoder │ (4 layers, d_model=256, 2 heads, kernel sizes [9, 1])
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Mel PostNet  │ (5-layer 1D Conv residual network)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   HiFi-GAN   │ (Universal Multi-Receptive Field Vocoder)
└──────┬───────┘
       │
       ▼
22.05 kHz Audio Waveform (.wav)
```

### Changes from ming024/FastSpeech2 Reference

1. **Hardware-Agnostic Execution**:
   Removed hardcoded `.cuda()` calls across `stft.py` and `modules.py`. All tensor allocations and transformations now track `input_data.device`, enabling execution on CPU, CUDA, and MPS without device mismatch errors.
2. **Modern Librosa Compatibility**:
   Updated deprecated positional calls in `librosa.filters.mel` and `pad_center` to keyword-only arguments for compatibility with Librosa >= 0.10.0.
3. **Phoneme-to-Frame Alignment**:
   Resolved punctuation token dropping issues in ARPAbet text cleaners to ensure strict length alignment across phonemes, duration vectors, pitch targets, and energy targets.
4. **Telugu Akshara G2P Engine (`core/text/telugu.py`)**:
   Built a native Telugu phonological front-end that parses Unicode text into Aksharas, decomposes consonant clusters, handles virama (halant) vowel deletion, binds dependent matras, maps anusvara ($ం \rightarrow \text{M}$), and expands Telugu numerals.
5. **Unified Configuration**:
   Consolidated separate configuration files into a single `config.yaml` per language for standalone execution.

---

## Datasets and Preprocessing

| Language | Dataset Source | Split | Audio Specifications |
| :--- | :--- | :--- | :--- |
| **English** | Svarah | 51 train / 9 val | 22,050 Hz, 16-bit PCM, mono |
| **Telugu** | IndicTTS Telugu | 51 train / 9 val | 22,050 Hz, 16-bit PCM, mono |

### Acoustic Feature Extraction Pipeline

- **Mel Spectrogram**: 1024-point FFT, 256-sample hop length, Hann window, 80 mel filterbanks (80 Hz to 7600 Hz).
- **Pitch (F0)**: Fast autocorrelation F0 tracker bounded between 50 Hz and 400 Hz for voiced frames, log-scale normalized using dataset statistics in `stats.json`.
- **Energy**: Frame-level L2 norm computed from linear STFT magnitudes, normalized using dataset statistics.
- **Durations**: Phoneme-level duration frame allocations matching the exact spectrogram length.

---

## Evaluation Sentences

The test utterances cover distinct phonetic, morphological, and prosodic challenges across both languages.

### English Utterances

| ID | Test Sentence Text | Linguistic and Acoustic Nuance |
| :--- | :--- | :--- |
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | 28-letter polysyllabic word (12 syllables: `an-ti-dis-es-tab-lish-ment-ar-i-an-ism`). Tests duration stability across long phoneme sequences. |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | Noun form `/ˈkɒn.dʌkt/`: Lexical stress on the initial syllable (`K AA1 N`) with higher pitch and duration. |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | Verb form `/kənˈdʌkt/`: Lexical stress shifts to the second syllable (`D AH1 K T`), reducing the initial syllable to a schwa. |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | Internal whitespace in `rhyth  m`. Preprocessed into syllabic rhotic segments without dropping tokens or breaking alignment. |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | Mixed sentence with timestamps (`10:45 AM`), dates (`12 November 2026`), and an embedded Telugu loanword (`andaroo` = *everyone*). |

### Telugu Utterances

| ID | Test Sentence Text | Linguistic and Acoustic Nuance |
| :--- | :--- | :--- |
| **T1** | న్యూ రోఫార్మ కోలాజికల్  పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని  విశ్లేషించారు. | Long compound loanword (*neuropharmacological*) containing complex conjuncts (`/nʲuː/`, `/rk/`). |
| **T2a** | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | Noun *pattu* (*grip*): Retroflex geminate `/ʈʈ/` preceded by short vowel `/a/`, producing a stop closure and burst. |
| **T2b** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చ ింది . | Attributive noun *pattu* (*silk*) followed by palatal geminate `/tt͡ʃ/` in *vachindi*. |
| **T3** | పులి అడ  విలో నిశ్శ బ్దంగా నడిచింది. | Intra-word space in *ada  vilo* (*forest*) and sibilant geminate `/ɕɕ/` in *nisshabdamga* (*quietly*). |
| **T4** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | Formal administrative sentence with Telugu numeral expansions (`2026`, `12వ`, `10:45`). |

---

## Setup and Usage

### 1. Environment Installation

```bash
git clone <repository_url>
cd project

# Fetch large model checkpoints via Git LFS
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

### 2. Acoustic Preprocessing

Preprocess raw audio and transcriptions into mel-spectrograms, pitch, energy, durations, and `stats.json`:

```bash
# English
python English/preprocess.py

# Telugu
python Telugu/preprocess.py
```

### 3. Model Training

Train the FastSpeech 2 network:

```bash
# Train English model
python English/train.py

# Train Telugu model
python Telugu/train.py
```

Checkpoints are saved to:
- `English/checkpoints/checkpoint_best.pth.tar` and `checkpoint_final.pth.tar`
- `Telugu/checkpoints/checkpoint_best.pth.tar` and `checkpoint_final.pth.tar`

Loss curves are output to `Results/English/` and `Results/Telugu/`.

### 4. Synthesis and Inference

Generate 22.05 kHz wav files and variance plots for test sentences:

```bash
# Synthesize English sentences (E1, E2a, E2b, E3, E4)
python English/inference.py

# Synthesize Telugu sentences (T1, T2a, T2b, T3, T4)
python Telugu/inference.py
```

Outputs are saved to:
- Waveforms: `Results/*/test_generated_audios/*.wav`
- Variance plots: `Results/*/variance_adaptor_plots/{duration, pitch, energy}/*.png`

---

## Training and Loss Analysis

The model optimizes a multi-task objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{postnet}} + \mathcal{L}_{\text{duration}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}}$$

- **Mel & PostNet Loss**: Mean Absolute Error ($L_1$) on 80-channel mel-spectrogram targets.
- **Duration Loss**: Mean Squared Error ($L_2$) in log domain.
- **Pitch & Energy Loss**: Mean Squared Error ($L_2$) on normalized continuous contours.

Loss curves show stable convergence without gradient explosion or overfitting:
- English training and validation plots: `Results/English/training_loss.png`, `Results/English/validation_loss.png`
- Telugu training and validation plots: `Results/Telugu/training_loss.png`, `Results/Telugu/validation_loss.png`

---

## Variance Adaptor Observations

1. **Duration Predictor**:
   - **English**: High variance between stressed vowels (6 to 10 frames, ~70 to 116 ms) and unstressed vowels / stops (2 to 3 frames, ~23 to 35 ms).
   - **Telugu**: Syllable-timed rhythm shows uniform Akshara baseline durations, with duration peaks concentrated on geminates (e.g., `/ʈʈ/` in *pattu*).
2. **Pitch (F0) Predictor**:
   - Accurately captures the lexical stress shift in *conduct* (noun E2a initial syllable peak vs. verb E2b second syllable peak).
   - Telugu sentences display smooth phrase-level declination with boundary rises.
3. **Energy Predictor**:
   - Aligns with voicing states, dropping near zero during stop closures and rising during vowel nuclei.

---

## Checkpoints and Git LFS

Model checkpoints are tracked with Git LFS to comply with file size limits:
- `English/checkpoints/checkpoint_best.pth.tar`
- `English/checkpoints/checkpoint_final.pth.tar`
- `Telugu/checkpoints/checkpoint_best.pth.tar`
- `Telugu/checkpoints/checkpoint_final.pth.tar`
- `core/hifigan/generator_universal.pth.tar`

If Git LFS bandwidth is constrained, models can be trained directly using the training scripts above.

---

## Technical Report

The accompanying technical report (`Technical_Report.pdf`) contains full theoretical derivations, phonetic transcriptions, variance adaptor contour figures, and cross-lingual analysis. It is submitted directly via the course portal alongside this repository.
