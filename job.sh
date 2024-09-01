#!/bin/bash -l
#SBATCH --job-name=emb          # Job name
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:a100:1             # number of gpus per node
#SBATCH --ntasks=1
#SBATCH --mem=256GB 
#SBATCH --cpus-per-task=12
#SBATCH                              # This is an empty line to separate Slurm directives from the job commands

#run your job
source ~/miniforge3/etc/profile.d/conda.sh
conda activate emb 


cd /home/tnguyenho/workspace/job-classification
python run.py


#Run a Matlab script with parameters: $SLURM_ARRAY_TASK_ID, $alpha, and $beta, and then exit