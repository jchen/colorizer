#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 2
#SBATCH --time 48:00:00
#SBATCH --mem 96G
#SBATCH -p gpu --gres=gpu:1
#SBATCH -C geforce3090
#SBATCH --job-name cs1430_final_train
#SBATCH --output cs1430_final_colorizer-%J.txt
# Loads needed modules
module load python/3.9.0
module load cuda
source ~/Projects/cs1430/cs1430_env/bin/activate

# Goes into notebooks and executes Python
python run.py
