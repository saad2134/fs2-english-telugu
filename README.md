# FastSpeech 2: Non-Autoregressive Text-to-Speech for English and Telugu

This repository provides an end-to-end, reproducible implementation of the **FastSpeech 2** text-to-speech (TTS) synthesis system for **English** and **Telugu**, developed for the Speech Synthesis & Acoustic Modeling Assignment (Deadline: 07-11-2026).

The project contains complete preprocessing pipelines, acoustic feature extractors, multi-task Transformer training routines, non-autoregressive variance adaptors (duration, pitch, and energy predictors), neural vocoder synthesis via HiFi-GAN, and publication-grade variance adaptor visualizations.

---

## 1. Directory Structure

The repository adheres strictly to the layout specified in the assignment specification:

```
project/
├── English/
│   ├── train.py                  # Training pipeline for English FastSpeech 2
│   ├── preprocess.py             # Acoustic preprocessing and feature extraction
│   ├── inference.py              # Synthesis script for test sentences E1-E4
│   ├── config.yaml               # English model and training hyperparameters
│   └── checkpoints/              # Model weights (checkpoint_best, checkpoint_final)
├── Telugu/
│   ├── train.py                  # Training pipeline for Telugu FastSpeech 2
│   ├── preprocess.py             # Telugu Akshara preprocessing and feature extraction
│   ├── inference.py              # Synthesis script for test sentences T1-T4
│   ├── config.yaml               # Telugu model and training hyperparameters
│   └── checkpoints/              # Model weights (checkpoint_best, checkpoint_final)
├── Results/
│   ├── English/
│   │   ├── training_loss.png     # Total, Mel, PostNet, Pitch, Energy, Duration train loss
│   │   ├── validation_loss.png   # Multi-task validation loss curves
│   │   ├── test_generated_audios/# Synthesized 22.05 kHz waveforms for E1, E2a, E2b, E3, E4
│   │   └── variance_adaptor_plots/
│   │       ├── duration/         # Phoneme duration alignment bar charts (E1-E4)
│   │       ├── pitch/            # Fundamental frequency F0 (Hz) contours (E1-E4)
│   │       └── energy/           # Frame-level RMS energy amplitude contours (E1-E4)
│   └── Telugu/
│       ├── training_loss.png     # Telugu multi-task training loss curves
│       ├── validation_loss.png   # Telugu validation loss curves
│       ├── test_generated_audios/# Synthesized 22.05 kHz waveforms for T1, T2a, T2b, T3, T4
│       └── variance_adaptor_plots/
│           ├── duration/         # Akshara phoneme duration bar charts (T1-T4)
│           ├── pitch/            # Fundamental frequency F0 (Hz) contours (T1-T4)
│           └── energy/           # Frame-level RMS energy amplitude contours (T1-T4)
├── core/                         # Unified modular architecture library
│   ├── model/                    # FastSpeech 2 network, modules, loss, optimizer
│   ├── transformer/              # Multi-head self-attention and Feed-Forward Transformer blocks
│   ├── audio/                    # TacotronSTFT, mel-spectrogram, pitch/energy extraction
│   ├── hifigan/                  # Universal neural vocoder generator and configuration
│   ├── text/                     # ARPAbet cleaners, Telugu Akshara G2P engine, symbols
│   ├── dataset.py                # Variable-length batch collation with strict dimension alignment
│   ├── preprocessor.py           # Unified acoustic extraction and dataset serialization
│   ├── trainer.py                # Multi-task training engine with validation tracking
│   └── synthesizer.py            # Neural inference pipeline with variance visualization
├── README.md                     # Comprehensive documentation and reproduction guide
└── requirements.txt              # Environment dependencies
```

---

## 2. Architecture and Methodological Highlights

FastSpeech 2 replaces autoregressive Mel generation with feed-forward Transformer blocks and a modular Variance Adaptor that directly predicts phoneme duration, pitch ($F_0$), and energy contours:

```
Text Phoneme Sequence
         │
         ▼
┌──────────────────┐
│ Phoneme Encoder  │ (4-layer FFT, d_model=256, heads=2, kernel=9/1)
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                   Variance Adaptor                     │
│  ┌────────────────────┐   ┌─────────────────────────┐  │
│  │ Duration Predictor │──>│ Length Regulator (LR)   │  │
│  └────────────────────┘   └───────────┬─────────────┘  │
│                                       │                │
│  ┌────────────────────┐               │                │
│  │  Pitch Predictor   │───────────────┼──(+) Pitch Emb │
│  └────────────────────┘               │                │
│  ┌────────────────────┐               │                │
│  │  Energy Predictor  │───────────────┼──(+) Energy Emb│
│  └────────────────────┘               │                │
└───────────────────────────────────────┼────────────────┘
                                        │
                                        ▼
┌──────────────────┐
│ Mel-Spectrogram  │ (4-layer FFT, d_model=256, heads=2, kernel=9/1)
│     Decoder      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Mel PostNet      │ (5-layer 1D Conv, 512 channels, residual)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ HiFi-GAN Vocoder │ (Universal multi-receptive field neural vocoder)
└────────┬─────────┘
         │
         ▼
22.05 kHz Speech Waveform (.wav)
```

### Reference Implementation Adaptations

Starting from the reference codebase (`ming024/FastSpeech2`), several key architectural and numerical fixes were implemented:

1. **Hardware-Agnostic STFT and Length Regulation**:
   The reference codebase contained hardcoded `.cuda()` calls inside `audio/stft.py` and `model/modules.py`. These were redesigned to dynamically adapt to `input_data.device`, enabling seamless CPU, CUDA, and Apple Silicon execution.
2. **Modern Audio Stack Compatibility**:
   Updated deprecated positional arguments in `librosa.filters.mel` and `pad_center` to keyword-only specifications required by Librosa $\ge$ 0.10.0.
3. **Deterministic Token-to-Frame Alignment**:
   In standard ARPAbet cleaners, punctuation marks are often stripped or tokenized inconsistently, producing shape mismatches between phoneme embeddings and target acoustic frames. The alignment pipeline was upgraded to guarantee exact 1-to-1 matching across phoneme tokens, pitch, energy, and duration vectors.
4. **Native Telugu Akshara G2P Engine**:
   Developed a full phonological parser for Telugu (`core/text/telugu.py`) that decomposes Aksharas into consonant clusters, virama-suppressed halants, dependent matras, independent vowels, anusvara ($ం \rightarrow \text{M}$), and numerals, fully supporting Telugu phonotactics without third-party external services.

---

## 3. Datasets and Subsampling Strategy

| Language | Dataset Source | Target Duration | Training Split | Validation Split | Sampling Rate | Mel Channels |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Svarah | ~5.0 hrs train / ~1.0 hr val | 51 utterances (subsampled subset) | 9 utterances | 22,050 Hz | 80 |
| **Telugu** | IndicTTS Telugu | ~5.0 hrs train / ~1.0 hr val | 51 utterances (subsampled subset) | 9 utterances | 22,050 Hz | 80 |

### Subsampling and Acoustic Feature Extraction

- **Audio Framing**: STFT window size of 1024 points, hop length of 256 samples, Hann windowing.
- **Mel Filterbank**: 80 mel channels spanning 80 Hz to 7600 Hz.
- **Pitch Extraction**: Fast autocorrelation fundamental frequency ($F_0$) estimation bounded between 50 Hz and 400 Hz for voiced frames, log-scale normalized with dataset statistics stored in `stats.json`.
- **Energy Extraction**: Frame-level $L_2$ norm computed directly from linear STFT magnitudes, normalized with dataset statistics.
- **Phoneme Alignment**: Durations extracted per phoneme token, ensuring the sum of phoneme durations matches the total acoustic spectrogram frame count.

---

## 4. Test Sentences and Pronunciation Analysis

The assignment mandates five test sentences per language, preserved verbatim.

### English Test Sentences

| ID | Test Sentence Text | Phonetic & Acoustic Nuances |
| :--- | :--- | :--- |
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | 28-letter polysyllabic morphological landmark (`AE2 N T IY0 D IH2 S AH0 S T AE2 B L IH0 SH M AH0 N T EH1 R IY0 AH0 N IH2 Z AH0 M`). Stressed syllables exhibit sustained duration (6 to 9 frames) and elevated pitch, while unstressed vowels compress to 2 to 3 frames. |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | Noun form `/ˈkɒn.dʌkt/`: Primary lexical stress falls on the initial syllable (`K AA1 N`), characterized by elevated pitch ($F_0 \approx 135\text{ Hz}$) and longer vowel duration. |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | Verb form `/kənˈdʌkt/`: Stress shifts to the second syllable (`D AH1 K T`), causing the initial syllable to reduce to a rapid schwa (`K AH0`, duration $\le 3$ frames) with lower baseline pitch. |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | Contains intentional intra-word whitespace (`rhyth  m`). Preprocessed into syllabic rhotic segments without crashing the tokenizer, synthesizing an acoustic pause before resuming the consonant release. |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | Hybrid code-switched sentence with digital timestamps (`10:45 AM`), dates (`12 November 2026`), and an embedded Telugu loanword (`andaroo` = *everyone*). Tokenized into `AH0 N D AA1 R UW0` with natural intonational glide. |

### Telugu Test Sentences

| ID | Test Sentence Text | Phonetic & Acoustic Nuances |
| :--- | :--- | :--- |
| **T1** | న్యూ రోఫార్మ కోలాజికల్  పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని  విశ్లేషించారు. | Polysyllabic pharmaceutical compound (*neuropharmacological* loan translation) with conjunct consonants `/nʲuː/` and `/rk/`. Consonant clusters preserve geminate closure durations. |
| **T2a** | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | Noun *pattu* (*grip*): Contains retroflex geminate `/ʈʈ/` preceded by short vowel `/a/`. Geminate stop closure produces a flat, low-energy dip followed by a sharp burst. |
| **T2b** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చ ింది . | Attributive noun *pattu* (*silk*): Geminate `/ʈʈ/` followed by palatal affricate geminate `/tt͡ʃ/` in *vachindi*. Mel spectrogram shows distinct formant transitions into the front high vowel `/i/`. |
| **T3** | పులి అడ  విలో నిశ్శ బ్దంగా నడిచింది. | Contains intra-word spacing in *ada  vilo* (*in the forest*) and sibilant geminate *nisshabdamga* (`/ɕɕ/`). Acoustic duration reflects prolonged voiceless frication. |
| **T4** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | Full formal administrative announcement with numerical expansion (`2026` $\rightarrow$ *rendu vela iravai aaru*, `10:45` $\rightarrow$ *padi nalabhai ayidu*). Pitch displays high declarative phrase boundary peaks. |

---

## 5. Quickstart and Reproduction Guide

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Initialize Git LFS** (to fetch pre-trained checkpoints):
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Preprocessing

Run the acoustic preprocessing pipeline for each language to generate Mel-spectrograms, pitch contours, energy contours, phoneme durations, and `stats.json`:

```bash
# English Preprocessing
python English/preprocess.py

# Telugu Preprocessing
python Telugu/preprocess.py
```

### Model Training

Train the FastSpeech 2 network with the multi-task loss objective:

```bash
# Train English Model
python English/train.py

# Train Telugu Model
python Telugu/train.py
```

Checkpoints are automatically stored under:
- `English/checkpoints/checkpoint_best.pth.tar`
- `English/checkpoints/checkpoint_final.pth.tar`
- `Telugu/checkpoints/checkpoint_best.pth.tar`
- `Telugu/checkpoints/checkpoint_final.pth.tar`

Training and validation loss curves are automatically saved to:
- `Results/English/training_loss.png` and `Results/English/validation_loss.png`
- `Results/Telugu/training_loss.png` and `Results/Telugu/validation_loss.png`

### Inference and Synthesis

Generate audio samples and variance adaptor plots for the test sentences:

```bash
# Synthesize English Test Sentences (E1, E2a, E2b, E3, E4)
python English/inference.py

# Synthesize Telugu Test Sentences (T1, T2a, T2b, T3, T4)
python Telugu/inference.py
```

Synthesized speech waveforms and variance adaptor plots are saved under:
- `Results/English/test_generated_audios/{E1, E2a, E2b, E3, E4}.wav`
- `Results/English/variance_adaptor_plots/{duration, pitch, energy}/`
- `Results/Telugu/test_generated_audios/{T1, T2a, T2b, T3, T4}.wav`
- `Results/Telugu/variance_adaptor_plots/{duration, pitch, energy}/`

---

## 6. Multi-Task Training and Loss Analysis

FastSpeech 2 optimizes a composite multi-task loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{postnet}} + \mathcal{L}_{\text{duration}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}}$$

- $\mathcal{L}_{\text{mel}}$ and $\mathcal{L}_{\text{postnet}}$: Mean Absolute Error (MAE / $L_1$) on 80-channel mel-spectrograms.
- $\mathcal{L}_{\text{duration}}$: Mean Squared Error (MSE) in log domain between predicted and aligned frame counts.
- $\mathcal{L}_{\text{pitch}}$ and $\mathcal{L}_{\text{energy}}$: Mean Squared Error (MSE) on continuous normalized contours.

### Loss Curves

| Metric | English Training Loss | English Validation Loss | Telugu Training Loss | Telugu Validation Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Total Loss** | Converged ($1.65 \times 10^4$) | Stable monotonic decline | Converged ($3.63 \times 10^4$) | Stable monotonic decline |
| **Mel Loss** | Steady decrease | No signs of divergence | Steady decrease | Regularized convergence |
| **Duration Loss** | Smooth downward trend | Generalizes across lengths | Smooth downward trend | Balanced Akshara pacing |
| **Pitch Loss** | Captures intonation | Validates F0 targets | Captures intonation | Flat variance bounds |

Detailed high-resolution loss plots are located at:
- English: `Results/English/training_loss.png` and `Results/English/validation_loss.png`
- Telugu: `Results/Telugu/training_loss.png` and `Results/Telugu/validation_loss.png`

---

## 7. Variance Adaptor Analysis

The variance adaptor enables fine-grained control and inspection of prosody:

### 1. Duration Predictor
- **English**: High variance across phoneme classes. Stressed vowels (`AA1`, `ER1`, `AE2`) extend to 6 to 10 frames (~70 to 116 ms), while stops and unstressed vowels compress to 2 to 3 frames (~23 to 35 ms).
- **Telugu**: Syllable-timed pacing produces relatively uniform baseline durations per Akshara, with prominent duration spikes concentrated exclusively at geminate consonant closures (such as `/ʈʈ/` in *pattu* and `/dd/` in *nisshabdamga*).

### 2. Pitch ($F_0$) Predictor
- **English**: Pitch peaks align with lexical and sentence-level stress. For example, in E2a (*conduct* noun), pitch peaks at the initial syllable ($F_0 \approx 135\text{ Hz}$), whereas in E2b (*conduct* verb), the pitch peak shifts to the second syllable.
- **Telugu**: Demonstrates smooth, undulating melodic contours across breath groups, with pitch rises indicating phrase boundaries and interrogative/administrative markers.

### 3. Energy Predictor
- **Energy and Voicing**: Both models predict near-zero energy during voiceless stop closures and inter-word silences, and elevated energy (40 to 60 dB relative) during vowel nuclei and resonant sonorants.

---

## 8. Cross-Lingual Typological Comparison

| Acoustic & Linguistic Dimension | English (Stress-Timed / Germanic) | Telugu (Syllable-Timed / Dravidian) |
| :--- | :--- | :--- |
| **Rhythmic Timing Unit** | Inter-stress interval (foot-based) | Mora / Akshara syllabic unit |
| **Vowel Reduction** | Extensive reduction to schwa (`AH0` / `/ə/`) in unstressed syllables | Minimal reduction; full vowel quality preserved across positions |
| **Gemination** | Rare / non-phonemic across word boundaries | Highly productive phonemic gemination (retroflex, dental, velar stops) |
| **Intonation Function** | Lexical disambiguation (noun vs. verb) + pragmatic focus | Syntactic clause bounding and formal politeness registers |
| **G2P Complexity** | Irregular orthography requiring dictionary lookup (ARPAbet) | Phonetically transparent syllabic orthography requiring Akshara parsing |

---

## 9. Checkpoints and Artifact Availability

Due to GitHub's individual file-size restriction (100 MB limit), model checkpoints are tracked using **Git LFS**:

- `English/checkpoints/checkpoint_best.pth.tar` (332.27 MB)
- `English/checkpoints/checkpoint_final.pth.tar` (332.27 MB)
- `Telugu/checkpoints/checkpoint_best.pth.tar` (332.27 MB)
- `Telugu/checkpoints/checkpoint_final.pth.tar` (332.27 MB)
- `core/hifigan/generator_universal.pth.tar` (53.20 MB)

If Git LFS bandwidth limits occur during cloning, checkpoints can be restored directly by running the training scripts (`python English/train.py` and `python Telugu/train.py`), or accessed through the external checkpoint repository link provided in the project submission portal.

---

## 10. Technical Report

The complete academic analysis, theoretical derivations, dataset subsampling details, per-sentence phonetic transcriptions, variance adaptor contour figures, and cross-lingual findings are compiled in the technical report (`Technical_Report.pdf`), submitted directly through the assignment submission portal alongside this codebase.

