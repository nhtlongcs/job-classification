#!/bin/bash -l
#SBATCH --job-name=llm         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:a100:1             # number of gpus per node
#SBATCH --time=1-00:00:00          # total run time limit (HH:MM:SS)

#run your job

# REQUIRED
# udocker create --name=llm lmsysorg/sglang:v0.3.2-cu121
export PATH=/home/tnguyenho/setup/udocker-1.3.16/udocker:$PATH
export MODEL_ARG=/home/tnguyenho/workspace/shared-llm/gemma-2-27b-it
export CONTAINER_NAME=llm

udocker run -p 30000:30000 -v $MODEL_ARG:/sgl-workspace/llm $CONTAINER_NAME python3 -c "import torch; print(f'TORCH_VERSION={torch.__version__}\nCUDA_AVAILABLE={torch.cuda.is_available()}\nTORCH_CUDA_ARCH_LIST={torch.cuda.get_device_capability()}')"
udocker run -p 30000:30000 -v $MODEL_ARG:/sgl-workspace/llm $CONTAINER_NAME python3 -m sglang.launch_server  --model-path /sgl-workspace/llm --host 0.0.0.0 --port 30000 --disable-cuda-graph
