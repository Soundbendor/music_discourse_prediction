#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --job-name=music_discourse_prediction
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --cpus-per-task 2
#SBATCH --export=ALL

source /nfs/guille/eecs_research/soundbendor/beerya/miniconda3/bin/activate 
source activate mdp 
bert_features --source Reddit Youtube Twitter --dataset amg1608 --intersect True --batch_size 64 --epochs 2 --model bert-base-uncased
