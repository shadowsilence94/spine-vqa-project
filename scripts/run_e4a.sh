#!/bin/bash
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=E4a
#SBATCH --output=/home/dsia-st125985/SpineVQA/logs/e4a_%j.log
PYTHON=/home/dsia-st125985/.conda/envs/spinevqa/bin/python
$PYTHON ~/SpineVQA/scripts/E4_ablations.py --model E4a
