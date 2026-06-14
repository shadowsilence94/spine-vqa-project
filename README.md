# SpineVQA: Localization-Aware Multimodal Learning for Spinal Disease Question Answering

This repository contains the source code and implementation for the group project **SpineVQA**, targeting the **SpineBench** dataset.

## Project Structure
- `main.py`: Entry point for training, validation, and evaluation.
- `requirements.txt`: Python package requirements.
- `src/`:
  - `dataset.py`: PyTorch Dataset class for parsing the SpineBench dataset (X-ray images, clinical questions, answers, and localization levels).
  - `model.py`: Multimodal fusion network combining vision encoders (SigLIP/CLIP/ViT) and text encoders (BERT/PubMedBERT) with a joint multitask classification head.
  - `train.py`: Multitask training loop with inverse frequency class weighting and contrastive alignment.
  - `eval.py`: Performance metrics computation (VQA accuracy, localization accuracy, macro F1, and confusion matrix).

## Features
- **Multimodal Fusion**: Direct feature fusion of visual embeddings (from Vision Transformers/SigLIP) and text embeddings (from PubMedBERT).
- **Multitask Optimization**: Jointly optimizes for VQA answer prediction, disease classification, and vertebral-level lesion localization.
- **Metal Performance Shaders (MPS)**: Native Apple Silicon GPU acceleration.
- **Class Imbalance Mitigation**: Computes and applies inverse class weights for training stability on highly imbalanced spine disease categories.

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run baseline training (Dry run with synthetic dataset):
   ```bash
   python main.py --dry-run
   ```
