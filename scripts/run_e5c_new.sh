#!/bin/bash
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=E5c
#SBATCH --output=/home/dsia-st125985/SpineVQA/logs/e5c_%j.log
PYTHON=/home/dsia-st125985/.conda/envs/spinevqa/bin/python
$PYTHON ~/SpineVQA/scripts/E5c_freeze_bert.py
