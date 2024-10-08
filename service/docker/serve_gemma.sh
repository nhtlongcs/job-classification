export MODEL_ARG=/home/tnguyenho/workspace/shared-llm/gemma-2-27b-it

docker run --gpus all \
    -v $MODEL_ARG:/sgl-workspace/llm lmsysorg/sglang:latest python3 -m sglang.launch_server  --model-path /sgl-workspace/llm --host 0.0.0.0 --port 30000
