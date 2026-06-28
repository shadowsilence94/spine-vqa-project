#!/bin/bash
#SBATCH --partition=ASL-solar
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=E4c
#SBATCH --output=/home/dsia-st125985/SpineVQA/logs/e4c_%j.log
PYTHON=/home/dsia-st125985/.conda/envs/spinevqa/bin/python
$PYTHON ~/SpineVQA/scripts/E4_ablations.py --model E4c
