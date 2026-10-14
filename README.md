---
title: Ayodo Face Type
emoji: 🧑
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.43.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Ayodo Face Type

Ayodo Face Type is a simple Hugging Face Spaces app for **face type classification**, not person identification. It accepts one face image from upload or webcam, checks that the photo is usable, and returns a best-match face-shape label.

The face-shape label is **not** an identity, authentication, or tokenization signal and must not be used as one. A 12-label taxonomy carries at most 3.58 bits of entropy, and the real distribution is heavily skewed, so two unrelated people collide on the same label frequently. The label is also derived from immutable facial morphology, so it cannot be revoked after compromise, and face-shape morphometrics correlate with ancestry. This app is a face-shape classification demo only.

This app uses interpretable facial geometry inspired by the feature families discussed in IBM's *Diversity in Faces* work, then maps those measurements into a practical 12-label taxonomy:

- Oval
- Round
- Square
- Rectangle
- Oblong
- Diamond
- Heart
- Inverted Triangle
- Triangle
- Pear
- Tapered Oval
- Tapered Square

## Files

- `app.py`: Gradio app entry point for Hugging Face Spaces
- `face_type_pipeline.py`: detection, quality checks, alignment, geometry features, and 12-class prediction
- `models/`: YuNet face detector and LBF landmark model required for inference
- `requirements.txt`: minimal Python dependencies

## Local run and verification

For a server handoff, see [DEPLOY.md](DEPLOY.md). The ZIP includes the model files, so its recipient does not need Git LFS.

Use Python 3.11. This GitHub repository includes the two model files directly, so a normal clone is sufficient.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python app.py
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860`). Upload a clear image with one front-facing face and press **Submit**. The result contains a label and a heuristic match score; the score is not a calibrated probability.

The included smoke test exercises OpenCV 5 detection, landmark fitting, rotated and large photos, and rejection of a blank image. Its [NASA test photograph](tests/fixtures/README.md) provides a repeatable software check, not evidence of classification accuracy.

## Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Upload this repository with its Git LFS model files.
3. Keep `app.py` at the repo root.
4. Spaces will install `requirements.txt` and launch the app automatically.

## Notes

- The app is designed for one visible face at a time.
- It rejects blurry, distant, or strongly side-posed photos.
- It does not perform identity recognition, age prediction, gender prediction, or medical diagnosis.
- The 12 labels and thresholds are handcrafted. There is no labeled validation dataset, demographic evaluation, or calibrated confidence model in this repository. Do not use the output for access, payments, or identity decisions.
- This repository is the runnable face-shape demo. The credential verification, AWS deployment, and COOL benchmarks described in the separate competition proposal are not implemented here.
