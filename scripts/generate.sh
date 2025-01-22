#!/bin/bash

#SBATCH --job-name=una_generate
#SBATCH --gpus=1
#SBATCH --partition=rtx8000  

#SBATCH --mail-type=FAIL
#SBATCH --account=una_account  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4	
#SBATCH --mem=32G
#SBATCH --time=6-23:59:59
#SBATCH --output=~/results/logs/%j.log # Standard output and error log 

 

date;hostname;pwd
python ~/preprocess/anomaly_dataset.py  
date
