#!/bin/bash

#SBATCH --job-name=una_train
#SBATCH --gpus=1
#SBATCH --partition=rtx8000 

#SBATCH --mail-type=FAIL
#SBATCH --account=una_account  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G # 256G
#SBATCH --time=6-23:59:59
#SBATCH --output=~/results/logs/%j.log # Standard output and error log 


# exp-specific cfg #
gen_cfg_file=~/cfgs/generator/train/una_generator.yaml
train_cfg_file=~/cfgs/trainer/train/una_trainer.yaml 


date;hostname;pwd
python ~/scripts/train.py $gen_cfg_file $train_cfg_file 
date