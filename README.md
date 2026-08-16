# GreetSense - Smart-Face-Recogniser

**Real-time facial identity recognition with multi-model verification and live performance evaluation.**
<img width="1179" height="638" alt="image" src="https://github.com/user-attachments/assets/46216384-03ca-4f2d-af43-73000362a4e2" />


## Overview

GreetSense is a real-time facial recognition system that identifies individuals from a live video feed and verifies their identity using an ensemble of two independent deep learning models. 
Rather than relying on a single embedding model — which can behave unpredictably depending on lighting, angle, or image quality — GreetSense cross-validates identity using **FaceNet** and **ArcFace** in parallel, with per-model confidence thresholds tuned to each model's distance distribution.

The system was built to explore a practical question in applied computer vision: *how do you make a real-time recognition pipeline reliable enough to trust, given that any single model's confidence can be noisy frame-to-frame?* <br>

The answer implemented here combines model ensembling, temporal smoothing across frames, and a built-in evaluation harness to measure and compare model behavior empirically rather than assuming it.

## How It Works

```
Video frame → Face Detection (RetinaFace, OpenCV fallback)
           → Preprocessing (crop, resize, color-space normalization)
           → Parallel embedding extraction (FaceNet + ArcFace)
           → Cosine-distance matching against stored identity vectors
           → Per-model confidence thresholding
           → Temporal smoothing (rolling majority vote across recent frames)
           → Verified identity or "unrecognized"
```

Each stage is implemented as an independent, swappable component — detection, preprocessing, embedding, matching, and evaluation are all decoupled, so any piece (e.g. the detector backend, or the embedding models used) can be replaced without touching the rest of the pipeline.

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **Dual-model ensemble** (FaceNet + ArcFace) | Each model produces a differently-shaped embedding space; combining both reduces the failure modes of relying on one |
| **Cosine distance over Euclidean** | Scale-invariant, which matters for neural embeddings where magnitude isn't semantically meaningful |
| **Temporal smoothing (rolling window vote)** | Raw per-frame predictions are noisy; a short majority-vote window trades a small amount of latency for materially more stable identification |
| **RetinaFace with an OpenCV fallback** | RetinaFace is more accurate but not universally reliable across hardware/OS combinations; the fallback keeps the system functional everywhere |
| **Built-in evaluation harness** | Accuracy claims for a real-time system are meaningless without a way to measure them live, against live conditions — not just a static test set |

## Evaluation

GreetSense includes a live evaluation mode that computes, per model, in real time:
- Recognition accuracy
- Confusion matrix
- Average inference latency

This surfaced a concrete finding during testing: **ArcFace consistently outperformed FaceNet in accuracy and threshold stability** under identical conditions, while FaceNet's wider distance distribution made it more sensitive to threshold placement. This kind of empirical, side-by-side model comparison — rather than trusting a single model's reported benchmark numbers — is the core reason the evaluation module exists.

## Tech Stack

Python · OpenCV · DeepFace (RetinaFace, FaceNet, ArcFace) · Streamlit · scikit-learn · NumPy · Matplotlib

## Setup

**Requirements:** Python 3.10, a webcam

```bash
conda create -n greetsense python=3.10
conda activate greetsense
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. **Register** — enter a name and capture a face to store its reference embeddings.
2. **Recognize** — the system identifies registered individuals automatically and displays a live confidence-backed match.
3. **Evaluate** — enter a known identity, let the system run for a few seconds, and view per-model accuracy, confusion matrices, and latency.
