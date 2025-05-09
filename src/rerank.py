import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Plus
from llm.utils import convert_to_str
from nlp.bm25 import JobRetriever
from utils import get_config_section, read_config

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
    return parser.parse_args()


def load_configurations(config_path):
    cfg = read_config(config_path)
    model_cfg = get_config_section(cfg, "embedding")
    rerank_cfg = get_config_section(cfg, "reranker")
    enriched_data_cfg = get_config_section(cfg, "llm")
    return model_cfg, rerank_cfg, enriched_data_cfg


def load_data(model_cfg, enriched_data_cfg):
    top_k_preds = pd.read_csv(model_cfg["output"], dtype=str)
    labels_set = pd.read_csv(enriched_data_cfg["labels"], dtype=str)
    top_k_preds = convert_to_str(top_k_preds)
    labels_set = convert_to_str(labels_set)
    return top_k_preds, labels_set


def rerank_predictions(row, labels_set, job_retriever):
    query = row["translated_title"]
    if pd.isnull(query):
        query = row["title"]
        
    top_k_preds = row["pred_code"].split(", ")
    top_k_titles = labels_set[labels_set["code"].isin(top_k_preds)][
        "english title"
    ].tolist()
    similar_job_titles = labels_set[labels_set["code"].isin(top_k_preds)][
        "title en"
    ].tolist()
    preprocessed_query = job_retriever._preprocess(query)
    preprocessed_titles = [
        job_retriever._preprocess(title) for title in top_k_titles
    ]
    preprocessed_similar_job_titles = [
        job_retriever._preprocess(title) for title in similar_job_titles
    ]

    corpus = [
        x + " " + y
        for x, y in zip(preprocessed_titles, preprocessed_similar_job_titles)
    ]
    try:
        bm25 = BM25Plus(corpus)
    except:
        import pdb; pdb.set_trace()
    scores = bm25.get_scores(preprocessed_query)

    reranked_indices = np.argsort(scores)[::-1][: len(top_k_preds)]
    reranked_preds = [top_k_preds[i] for i in reranked_indices]

    return reranked_preds


def main():
    args = parse_args()
    model_cfg, rerank_cfg, enriched_data_cfg = load_configurations(args.config)

    if rerank_cfg.get("bm25", False):
        top_k_preds, labels_set = load_data(model_cfg, enriched_data_cfg)
        job_retriever = JobRetriever(labels_set=labels_set)

        reranked_preds = top_k_preds.progress_apply(
            lambda row: rerank_predictions(row, labels_set, job_retriever),
            axis=1,
        )

        top_k_preds["reranked_preds"] = reranked_preds.apply(
            lambda x: ", ".join(x)
        )
        top_k_preds.to_csv(rerank_cfg["output"], index=False)
    else:
        print("BM25 reranking is disabled")


if __name__ == "__main__":
    main()
