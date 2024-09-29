# Web Intelligence - Classification Challenge

This repository contains the code of team **FVNWL** for the European Statistics Awards' Web Intelligence Classification Challenge, focusing on classifying job advertisements into predefined occupational categories.

## Overview

The challenge aims to develop methodologies for processing web content to extract valuable data for statistical and analytical purposes. Specifically, this project focuses on assigning class labels from a known taxonomy to job advertisements.

## Key Components
The detailed description of our approach can be found in the report `classification_approach_description.docx`.

### Embedding and Semantic Retrieval
- `embedding/`: Contains scripts for text embedding
- `run.py`: Handles semantic retrieval operations

### External Data
- `enriched_data/`: Includes collected ISCO (International Standard Classification of Occupations) data from previous years

### LLM Inference
- `prompting.py`: Manages prompt engineering for the LLM
- `LLM-infer.py`: Handles inference using the Gemini LLM

### Utility and Submission
- `utils.py`: Contains utility functions for data preprocessing, language detection, and abbreviation handling
- `submission.py`: Parses LLM outputs and generates the submission file

### Ensemble Model
- `ensemble.py`: Implements digit-wise ensembling for improved classification results

## Contributors

- [@tqtnk2000](https://github.com/tqtnk2000)
- [@nhtlongcs](https://github.com/nhtlongcs)
- [@vantuan5644](https://github.com/vantuan5644)
- [@HongHanh2104](https://github.com/HongHanh2104)
