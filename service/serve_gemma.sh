#!/bin/bash -l
#SBATCH --job-name=llm         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:a100:1             # number of gpus per node
#SBATCH --time=2-00:00:00          # total run time limit (HH:MM:SS)

#run your job

# REQUIRED
# udocker create --name=llm lmsysorg/sglang:latest
export PATH=/home/tnguyenho/setup/udocker-1.3.16/udocker:$PATH
export MODEL_ARG=/home/tnguyenho/workspace/shared-llm/gemma-2-27b-it

udocker run -p 30000:30000 -v $MODEL_ARG:/sgl-workspace/gemma llm python3 -m sglang.launch_server  --model-path /sgl-workspace/gemma --host 0.0.0.0 --port 30000
