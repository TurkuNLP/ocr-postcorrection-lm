#!/bin/bash
#SBATCH --job-name=mixtral-test
#SBATCH --account=project_2000539
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module load pytorch

source ../.venv/bin/activate

python run_lm.py --input by_page_test_slim_sub_sample.jsonl.gz --out mixtral_output.jsonl

