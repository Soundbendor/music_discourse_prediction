#!/bin/bash
#SBATCH -w dgx2-5
#SBATCH -p dgx2
#SBATCH -A eecs
#SBATCH --job-name=music_discourse_prediction
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task 6
#SBATCH --export=ALL

source /nfs/guille/eecs_research/soundbendor/beerya/miniconda3/bin/activate 
source activate mdp 
bert_features --source Twitter --dataset amg1608 --intersect True --batch_size 128 --epochs 4
