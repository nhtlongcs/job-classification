import argparse
import os
import time
from collections import deque

import pandas as pd
from utils import TimedTQDM
from dotenv import load_dotenv

from llm.io import get_last_processed_row, compose_output
from llm.model import GenerativeModelWrapper
from llm.prompting import (
    generate_prompt,
    get_retrieved_info,
    get_system_prompt,
)
from llm.utils import switch_api_key

from utils import read_config, get_config_section

# Load environment variables
load_dotenv(override=True)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve job descriptions based on an embedding model"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the last processed row",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        help="Batch size for inference",
    )
    return parser.parse_args()


def load_configuration(config_path):
    cfg = read_config(config_path)
    llm_cfg = get_config_section(cfg, "llm")
    emb_cfg = get_config_section(cfg, "embedding")
    return llm_cfg, emb_cfg


def initialize_model(llm_cfg):
    system_prompt = get_system_prompt()
    if llm_cfg["name"] in ["gemma2", "mistralnemo"]:
        model = GenerativeModelWrapper(
            model_name="gemma2",
            model_version="002",
            system_instruction=system_prompt,
            host=os.getenv("GENAI_URL"),
            use_local=True,
        )
        print("Using local model")
        if os.getenv("GENAI_URL") is None:
            raise ValueError(
                "GENAI_URL environment variable not set, please set it to the local server URL"
            )
    else:
        model = GenerativeModelWrapper(
            model_name=llm_cfg["name"],
            model_version=llm_cfg["version"],
            system_instruction=system_prompt,
            host=os.getenv("GENAI_URL"),
            use_local=False,
        )
        api_keys = os.getenv("GENAI_API", None)
        print("Using Google GenAI API")
        if api_keys is not None:
            raise ValueError(
                "API keys not found in environment variables, please set GENAI_API=<API_KEY1>,.."
            )
    return model


def load_data(emb_cfg, llm_cfg):
    queried_data = pd.read_csv(emb_cfg["output"], dtype=str)
    labels = pd.read_csv(llm_cfg["labels"])
    return queried_data, labels


def process_job_ads(
    model, queried_data, labels, log_file_path, resume, api_keys, rate_limit=6, batch_size=1
):
    current_key, api_keys = switch_api_key(model, api_keys)
    last_processed_row = get_last_processed_row(log_file_path) if resume else 0
    print(f"Resuming from row: {last_processed_row}")

    write_mode = "a"
    if not resume:
        with open(log_file_path, "w") as log_file:
            pass

    pbar = TimedTQDM(range(last_processed_row, len(queried_data)))
    batch_prompts = []
    batch_indices = []

    for i in pbar:
        job_ad = queried_data.iloc[i]
        retrieved_job_info_txt = get_retrieved_info(
            job_ad, k=10, labels_df=labels, return_text=True
        )
        batch_prompts.append(generate_prompt(job_ad, retrieved_job_info_txt))
        batch_indices.append(i)

        if len(batch_prompts) == batch_size or i == len(queried_data) - 1:
            try:
                batch_response = model.generate_batch_content(
                    batch_prompts, generation_config={"temperature": 0}
                )
                if "localhost" in os.getenv("GENAI_URL") or "0.0.0.0" in os.getenv(
                    "GENAI_URL"
                ):
                    pass  # No rate limit for localhost
                else:
                    time.sleep(rate_limit)  # To avoid hitting rate limits

                with open(log_file_path, write_mode) as log_file:
                    for idx, response in zip(batch_indices, batch_response):
                        job_ad = queried_data.iloc[idx]
                        log_file.write(
                            compose_output(
                                job_ad.id, idx, job_ad.pred_code, response
                            )
                        )
                # Reset batch
                batch_prompts = []
                batch_indices = []

            except KeyboardInterrupt:
                print("Process interrupted by user. Exiting...")
                print(f"Processed {i} items.")
                break
    pbar.save_times(file_path="inference_time2.npy")
    print("Inference completed.")


def main():
    args = parse_args()
    llm_cfg, emb_cfg = load_configuration(args.config)
    model = initialize_model(llm_cfg)
    queried_data, labels = load_data(emb_cfg, llm_cfg)
    api_keys = deque(os.getenv("GENAI_API", "").split(","))
    process_job_ads(
        model, queried_data, labels, llm_cfg["output"], args.resume, api_keys, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
