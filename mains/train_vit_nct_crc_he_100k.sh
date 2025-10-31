#!/bin/bash

#SBATCH -J VIT_TRAIN
#SBATCH -p gpu
#SBATCH --mail-user=cwseitz@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=14:00:00
#SBATCH --gpus-per-node v100:4
#SBATCH --mem=512G
#SBATCH -A r01151

module load conda
conda activate spatial
python train_vit_nct_crc_he_100k.py
