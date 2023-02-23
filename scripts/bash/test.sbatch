#!/bin/env bash
#SBATCH --job-name=k1=15_k2=10
#SBATCH --output=/home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/output/cluster/impute_k1=15_k2=10.out
#SBATCH --error=/home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/output/cluster/impute_k1=15_k2=10.out
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8096
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
module load anaconda3
cd /home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/scripts/data_processing
eval "$(conda shell.bash hook)"
conda activate general

python choose_k.py 

