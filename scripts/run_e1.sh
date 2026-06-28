#!/bin/bash
#SBATCH --partition=ASL-lunar
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=E1_SpineVQA
#SBATCH --output=/home/dsia-st125985/SpineVQA/logs/e1_%j.log

# Conda PATH fix
export PATH="/home/dsia-st125985/.conda/bin:$PATH"
source /home/dsia-st125985/.conda/etc/profile.d/conda.sh
conda activate spinevqa

# Fix huggingface-hub version
pip install huggingface-hub==0.36.2 transformers==4.45.0 --quiet

cd /home/dsia-st125985/SpineVQA
python scripts/E1_baseline.py
