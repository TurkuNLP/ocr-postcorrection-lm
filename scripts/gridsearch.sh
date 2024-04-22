#!/bin/bash
#SBATCH --job-name=mixtral-test
#SBATCH --account=project_2005072
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=90G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module load pytorch

source ../.venv/bin/activate

python gridsearch.py --max_examples 10 --ntrials 10  --quantization $1