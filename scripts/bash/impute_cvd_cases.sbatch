#!/bin/env bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8096
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
module load anaconda3
cd /home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/scripts/data_processing
eval "$(conda shell.bash hook)"
conda activate general2

python impute_cvd.py ${K} ${percentage_missing}

