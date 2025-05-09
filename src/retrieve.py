import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from embedding.pipeline import OpenAIRetrieverPipeline, RetrieverPipeline
from embedding.utils import get_query_results
from utils import read_config, get_config_section

tqdm.pandas()


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
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="artifacts/top_k_prediction/",
        help="Path to the output file",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="Number of top k predictions",
    )
    return parser.parse_args()


def verify_config(cfg, args):

    if "dataset" not in cfg:
        raise ValueError("Missing dataset section in the configuration file")
    required_keys = ["ml_output", "en_output"]
    for key in required_keys:
        if key not in cfg["dataset"]:
            raise ValueError(
                f"Missing {key} in the dataset section of the configuration file"
            )

    output_dir = Path(args.output).parent
    if not output_dir.exists():
        raise ValueError("Output directory does not exist")
    Path(args.output).mkdir(exist_ok=True)


def get_model_factory(dataset_cfg):
    return {
        "text-embedding-3-large": (
            OpenAIRetrieverPipeline,
            dataset_cfg["ml_output"],
        ),
        "BAAI/bge-en-icl": (RetrieverPipeline, dataset_cfg["en_output"]),
        "dunzhang/stella_en_1.5B_v5": (
            RetrieverPipeline,
            dataset_cfg["en_output"],
        ),
        "Alibaba-NLP/gte-Qwen2-7B-instruct": (
            RetrieverPipeline,
            dataset_cfg["ml_output"],
        ),
        "BAAI/bge-multilingual-gemma2": (
            RetrieverPipeline,
            dataset_cfg["ml_output"],
        ),
    }


def main():
    args = parse_args()
    cfg = read_config(args.config)
    dataset_cfg = get_config_section(cfg, "dataset")
    model_cfg = get_config_section(cfg, "embedding")
    verify_config(cfg, args)

    top_k = args.top_k
    output_path = Path(args.output)

    model_factory = get_model_factory(dataset_cfg)
    model = model_cfg["name"]
    model_name = model.split("/")[-1]
    labels_filepath = model_cfg["labels"]
    retrieverPipeline, jd_filepath = model_factory[model]
    instruction = "Given a job advertisement, retrieve relevant job descriptions that matches the query."
    df_subset = pd.read_csv(jd_filepath)
    print(
        f"Indexing {len(pd.read_csv(labels_filepath))} job labels from {labels_filepath}"
    )

    pipeline = retrieverPipeline(
        labels_filepath, model=model, instruction=instruction
    )
    print(f"Retrieving top {top_k} job descriptions for each job description")
    df_result = df_subset.progress_apply(
        lambda row: pd.Series(
            get_query_results(pipeline, row["final_description"], top_k=top_k)
        ),
        axis=1,
    )
    df_result.columns = ["pred_code", "pred_label", "time"]
    processed_times = df_result["time"].tolist()
    import numpy as np
    mean_time = sum(processed_times) / len(processed_times)
    std_time = np.std(processed_times)

    print(f"Mean time: {mean_time}")
    print(f"Standard deviation: {std_time}")
    df_result = df_result[["pred_code", "pred_label"]]
    df_result = pd.concat([df_subset, df_result], axis=1)
    df_result.to_csv(
        output_path / f"classification_top_{top_k}_{model_name}.csv",
        index=False,
    )

    print(
        f"Precicted results saved to {output_path / f'classification_top_{top_k}_{model_name}.csv'}"
    )


if __name__ == "__main__":
    main()
