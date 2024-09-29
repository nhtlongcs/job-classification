
import rich
import random
from collections import deque
import time
import tqdm
import re
import os
import google.generativeai as genai
import pandas as pd
from prompting import generate_prompt, get_retrieved_info

# queried_data = pd.read_csv('top_k_prediction/classification_top_10_stella_en_1.5B_v5.csv')
queried_data = pd.read_csv(
    'top_k_prediction/classification_top_10_text-embedding-3-large_new.csv', dtype=str)
confidences = pd.read_csv('confidences.csv', dtype=str)
queried_data = queried_data.merge(confidences, on='id', how='left')
queried_data = queried_data.sort_values(
    by='Confidence', ascending=True).reset_index(drop=True)

labels = pd.read_csv(
    'datasets/EU-stats-challenge/submission-phase/wi_labels_enriched.csv')


system_prompt = (
    "As a chief human resources officer, you are tasked with analyzing job advertisements and determining the most appropriate ISCO unit for each job, based on the job description and job title. You have access to a list of potential ISCO units with their descriptions, definitions, and skill types. Analyze the job advertisement and the potential ISCO units, considering the main responsibilities and tasks described in the job ad, the required skills and qualifications, and the level of expertise and autonomy required. Provide a step-by-step reasoning process to determine the most appropriate ISCO unit for this job advertisement. Then, provide your final prediction in the format: ISCO Code: [code] ISCO Title: [title] Confidence: [0-1 scale] Reasoning: [A brief summary of your reasoning].")


model_name = 'gemini-1.5-flash'
model_version = '002'

model = genai.GenerativeModel(
    f"{model_name}-{model_version}", system_instruction=system_prompt)
generation_config = {
    "temperature": 0.25,
}

LOG_FILE = f'logs/{model_name}_log.txt'
API_KEYS = os.environ['GEMINI_API_KEYS'].split(',')
API_KEYS = deque(API_KEYS)
RESUME = True


def get_last_processed_row(log_file=LOG_FILE):
    if not os.path.exists(log_file):
        return 0

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if line.startswith("Job Ad ID:"):
                match = re.search(r"Row: (\d+)", line)
                if match:
                    return int(match.group(1)) + 1
    return 0


def switch_api_key(exhausted_key=None):
    global API_KEYS

    if exhausted_key:
        API_KEYS.remove(exhausted_key)
        API_KEYS.append(exhausted_key)

    if not API_KEYS:
        raise ValueError("All API keys have been exhausted")

    new_key = API_KEYS[0]
    genai.configure(api_key=new_key)

    print(f"Switched to a new API key: {new_key[:10]}...")

    return new_key

current_key = switch_api_key()

if RESUME:
    last_processed_row = get_last_processed_row()
    print(f"Resuming from row: {last_processed_row}")
else:
    last_processed_row = 0

with open(LOG_FILE, 'a') as f:
    i = last_processed_row
    pbar = tqdm.tqdm(total=len(queried_data) - last_processed_row)
    while i < len(queried_data):
        job_ad = queried_data.iloc[i]
        retrieved_job_info_txt = get_retrieved_info(
            job_ad, k=10, labels_df=labels, return_text=True)
        try:
            response = model.generate_content(generate_prompt(
                job_ad, retrieved_job_info_txt), generation_config=generation_config)
            f.write(
                f"Job Ad ID: {job_ad.id}, Row: {i}, Pred: {job_ad.pred_code}\n")
            f.write("Response:\n")
            f.write(f"{response.text}\n")
            f.write("\n\n")
            f.write(f"-----------------------------------\n\n")
            time.sleep(6)  # To avoid hitting rate limits
            i += 1
            pbar.update(1)
        except Exception as e:
            error_message = str(e)
            print(f"Error processing row {i}: {error_message}")
            current_key = switch_api_key(exhausted_key=current_key)
            time.sleep(10)  # Wait before retrying with new key
    pbar.close()

print("Inference completed.")
