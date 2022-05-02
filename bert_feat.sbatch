#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --job-name=music_emotion_pred
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:6
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/guille/eecs_research/soundbendor/beerya/miniconda3/bin/activate
source activate mdp
bert_features -i /nfs/guille/eecs_research/soundbendor/music_discourse/reddit/AMG1608 --source Reddit --dataset AMG1608 -m reddit_amg_model --num_epoch 50
