#!/bin/bash
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=E4_SpineVQA
#SBATCH --output=/home/dsia-st125985/SpineVQA/logs/e4_sky_%j.log

PYTHON=/home/dsia-st125985/.conda/envs/spinevqa/bin/python
PIP=/home/dsia-st125985/.conda/envs/spinevqa/bin/pip
$PIP install huggingface-hub==0.36.2 transformers==4.45.0 sentencepiece --quiet

cd /home/dsia-st125985/SpineVQA
$PYTHON scripts/E4_lavp_net.py
