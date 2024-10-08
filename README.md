# Web Intelligence Classification Challenge Repository

Welcome to the official repository of **Team FVNWL** for the European Statistics Awards' Web Intelligence Classification Challenge. This project focuses on the classification of job advertisements into standardized occupational categories, utilizing advanced web content processing techniques to extract valuable data for statistical analysis.

## 🚀 Project Overview

The challenge revolves around developing cutting-edge methodologies to process web content efficiently and accurately. Our primary goal is to assign class labels from a predefined taxonomy to job advertisements, facilitating more precise and scalable data extraction for statistical purposes. 

The project leverages state-of-the-art Large Language Models (LLMs), both locally and through APIs, to enhance the performance and reliability of our classification system.

For a deep dive into our approach, please refer to our comprehensive report: `classification_approach_description.docx`.

## 🛠️ Environment Setup

To set up the environment and reproduce our results, follow these steps:

### Step 1: Install Conda

Install Conda using the [Miniforge installation instructions](https://github.com/conda-forge/miniforge).

On Linux x86_64 (amd64), run the following commands to install Miniforge:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-Linux-x86_64.sh
sh Miniforge-pypy3-Linux-x86_64.sh
# Follow the on-screen instructions
```

### Step 2: Install Dependencies

Once Conda is installed, set up the required environment by running:

```bash
mamba env create -f env.yml
conda activate fvnwl
```

## 🔒 Data Access

Due to privacy and legal constraints, we are unable to share the original dataset used in this challenge. To facilitate reproducibility, we have synthesized part of the data for public use. This synthetic data mimics the structure and characteristics of the real data while ensuring compliance with privacy regulations.

If you would like to reproduce the full results with the original pre-run outputs, please contact us directly to request access to the necessary archived artifacts. Send an email to **[teamfvnwl@example.com](mailto:teamfvnwl@example.com)** with your request, and we will review it based on data-sharing agreements and guidelines.

## 💻 Usage

### Option 1: Reproduce Results with Pre-Run Outputs

If you'd like to reproduce our pre-run results, simply run:

```bash
sh scripts/run.sh
```

Ensure that:
- The `fvnwl` environment is activated.
- The `artifacts/archive` folder is available.

### Option 2: Run from Scratch

For those running the project from scratch, access to an LLM is required. We provide two options:

1. **Local LLM**: 
   - Set up the LLM locally by serving the model using the script in the `service` folder.
   - Use the provided `udocker` script if running on a Slurm cluster without root access.

2. **API LLM**:
   - Use an API to run the LLM model.
   - To handle API rate limits, provide multiple API keys by setting the `GENAI_API=<api_key_1>, <api_key_2>, ...` environment variable. The script will switch keys automatically when rate limits are reached.

### Testing the Environment

To confirm the environment setup, run:

```bash
sh scripts/test.sh
```

### Running Inference

Once the environment is ready, run the following command to execute inference:

```bash
sh scripts/inference.sh -c config/<config_file>.ini
```

Each configuration file in the `config` folder corresponds to specific results discussed in our report. Choose the appropriate configuration for your experiment.

## 🛡️ Contribution Guide

Interested in contributing? Here’s how to get started:

1. Fork the repository.
2. Create a new branch, e.g., `fix-loss`, `add-feat`.
3. Make your changes, add features, or fix bugs.
4. Add relevant test cases to the `test` folder.
5. Ensure all test cases pass (run `sh scripts/test.sh`).
6. Document your feature or bug fix in the Pull Request (PR).
7. Push your changes and submit a PR to the main repository.

Expected test results upon successful execution:

```bash
============================== test session starts ===============================

platform linux -- Python 3.10.15, pytest-8.3.3, pluggy-1.5.0 -- 
cachedir: .pytest_cache
rootdir: ...
configfile: pyproject.toml
plugins: anyio-4.6.0
collected 10 items   

tests/test_api_llm.py::test_generate_content PASSED                         [ 10%]
tests/test_api_llm.py::test_generate_content_with_different_prompt PASSED   [ 20%]
tests/test_api_llm.py::test_generate_content_with_high_temperature PASSED   [ 30%]
tests/test_keyword_extraction.py::test_keyword_extraction PASSED            [ 40%]
tests/test_langdet.py::test_detect_language PASSED                          [ 50%]
tests/test_local_llm.py::test_simple PASSED                                 [ 60%]
tests/test_local_llm.py::test_generate_content PASSED                       [ 70%]
tests/test_search.py::test_search_with_valid_description PASSED             [ 80%]
tests/test_search.py::test_search_with_special_characters PASSED            [ 90%]
tests/test_search.py::test_search_with_long_description PASSED              [100%]

============================== 10 passed in 35.13s ===============================
```

## 🌟 Contributors

A huge thanks to all contributors who made this project a success:

- [@tqtnk2000](https://github.com/tqtnk2000)
- [@nhtlongcs](https://github.com/nhtlongcs)
- [@vantuan5644](https://github.com/vantuan5644)
- [@HongHanh2104](https://github.com/HongHanh2104)

We look forward to your feedback and contributions to make this repository even better!