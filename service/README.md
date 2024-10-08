
# LLM Deployment Guide

Welcome to the deployment guide for our local LLM (Gemma 2 27B IT model)! This guide will walk you through the setup and deployment process using **udocker**, a user-friendly alternative to Docker that requires no root permissions. Below, you'll find a clear outline of each step required to get the LLM running on your server with GPU support.

---

## Why udocker?

Unlike traditional Docker, **udocker** is designed for environments where root permissions aren't available. It allows for containerized applications, such as our LLM model, to be run seamlessly without requiring admin-level access.

## Deployment Steps

### Step 1: Prepare the Environment

1. **Clone the repository**: Navigate to the `udocker` folder in this project and inspect the provided scripts.
2. **Install udocker**: Follow the included guide to install udocker on your machine. You can find the setup steps in the `udocker/README.md` file or refer to [udocker documentation](https://github.com/indigo-dc/udocker) for more details.

### Step 2: Create and Name Your Container

Once udocker is installed:

- **Create the container**: Use the provided command to create a new container image for the Gemma 2 27B IT model.
- **Name your container**: Assign a unique, descriptive name to the container for easy reference.

```bash
udocker pull lmsysorg/sglang:v0.3.2-cu121
udocker create --name=llm lmsysorg/sglang:v0.3.2-cu121
```

### Step 3: Grant GPU Permissions

After creating the container:

- **Enable GPU support**: Modify the container settings to grant GPU access, which is essential for running the large-scale LLM efficiently.

  You can configure GPU permissions using the following command:

```bash
udocker setup --nvidia llm
```

Make sure your system’s GPU drivers and CUDA are installed and configured properly.

### Step 4: Run the Container

Once the container is set up and the GPU access is enabled:

- **Run the LLM**: Finally, run the containerized LLM model by executing the following command:

```bash
udocker run -p 30000:30000 -v $MODEL_PATH:/sgl-workspace/llm $CONTAINER_NAME python3 -m sglang.launch_server  --model-path /sgl-workspace/llm --host 0.0.0.0 --port 30000

```

### Customization

You may need to adjust certain configurations or paths depending on your server environment. The scripts inside the `udocker` folder are designed to be flexible, but feel free to modify them according to your system's specifications.

### Troubleshooting

- If you encounter any issues related to dependencies or permissions, please refer to the udocker troubleshooting guide or the Gemma 2 27B IT model documentation for potential fixes.
