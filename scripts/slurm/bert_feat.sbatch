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
module load geant4
module load module load zlib-1.2.11-gcc-5.5.0-5rhntra

source /nfs/guille/eecs_research/soundbendor/beerya/miniconda3/bin/activate
source activate mdp

dataset=AMG1608
for src in reddit youtube twitter; do
    bert_features -i /nfs/guille/eecs_research/soundbendor/music_discourse/fix_data/$dataset/$src --source $src --dataset $dataset -m "${dataset}_${src}_model" --num_epoch 2
done

bert_features -i /nfs/guille/eecs_research/soundbendor/music_discourse/fix_data/$dataset/ --source "All" --dataset $dataset -m "${dataset}_All_model" --num_epoch 2
