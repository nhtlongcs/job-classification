import argparse
from pathlib import Path
import warnings
import pandas as pd
import rich
from llm.utils import (
    drop_duplicates_prioritize_non_null_with_confidence,
    is_effectively_null,
    read_and_parse_log,
    convert_to_str,
)
from nlp.lang import fuzzy_matching_df
from utils import read_config


def parse_log_to_dataframe(
    txt_logs_ls: list[list[str]], sources: list[str]
) -> pd.DataFrame:
    merged_df = pd.DataFrame()
    for txt_logs, source in zip(txt_logs_ls, sources):
        df = read_and_parse_log(txt_logs)
        df = convert_to_str(df)
        df["source"] = source
        rich.print(
            f"Source: {source} includes the following logs from {txt_logs}"
        )
        rich.print(f"DF shape: {df.shape}")
        rich.print(f"Number of unique Job Ad IDs: {df['Job Ad ID'].nunique()}")
        rich.print(f"Number of unique ISCO Codes: {df['ISCO Code'].nunique()}")
        rich.print(
            f"Number of not null ISCO Codes: {len(df[~df['ISCO Code'].apply(is_effectively_null)])}"
        )
        rich.print(
            f"Number of duplicated ids: {df['Job Ad ID'].duplicated().sum()}"
        )

        merged_df = pd.concat([merged_df, df])

    return merged_df


def load_and_prepare_cleaned_df(filepath: str) -> pd.DataFrame:
    cleaned_df = pd.read_excel(filepath, dtype=str)
    cleaned_df = cleaned_df[["id", "potential"]]
    cleaned_df.columns = ["Job Ad ID", "Alternative ISCO Codes"]
    cleaned_df["ISCO Code"] = cleaned_df["Alternative ISCO Codes"].apply(
        lambda x: x.split(", ")[0] if isinstance(x, str) else None
    )
    cleaned_df["Confidence"] = 1.0  # hard-coded confidence score
    cleaned_df["source"] = "cleaned"
    return cleaned_df


def merge_dataframes(df_list: list) -> pd.DataFrame:
    merged_df = pd.concat(df_list)
    merged_df = drop_duplicates_prioritize_non_null_with_confidence(
        merged_df,
        id_col="Job Ad ID",
        isco_code_col="ISCO Code",
        confidence_col="Confidence",
    )
    return merged_df


def load_submission_df(filepath: str) -> pd.DataFrame:
    submission_df = pd.read_csv(filepath)[["id"]]
    submission_df = convert_to_str(submission_df)
    return submission_df


def merge_with_submission(
    merged_df: pd.DataFrame, submission_df: pd.DataFrame
) -> pd.DataFrame:
    merged_df = merged_df.merge(
        submission_df, left_on="Job Ad ID", right_on="id", how="right"
    )
    return merged_df


def load_top_k_predictions(filepath: str) -> pd.DataFrame:
    top_k_preds = pd.read_csv(filepath, dtype=str)
    top_k_preds = convert_to_str(top_k_preds)
    return top_k_preds


def load_labels_set(filepath: str) -> pd.DataFrame:
    labels_set = pd.read_csv(filepath, dtype=str)
    labels_set = convert_to_str(labels_set)
    return labels_set


def handle_top_k_predictions(row, allowed_codes):
    top_k = (
        [
            code
            for code in row["pred_code"].split(", ")
            if code in allowed_codes
        ]
        if isinstance(row["pred_code"], str)
        else []
    )
    return top_k


def get_top_1_predictions(row):
    top_1 = row["top-k"][0] if row["top-k"] else None
    if "reranked_preds" in row and top_1 != row["ISCO Code"]:
        top_1 = row["reranked_preds"].split(", ")[0]
    return top_1.zfill(4) if top_1 else None


def handle_alternative_predictions(row, allowed_codes):
    top_k = (
        [
            code
            for code in row["Alternative ISCO Codes"].split(", ")
            if code in allowed_codes
        ]
        if isinstance(row["Alternative ISCO Codes"], str)
        else []
    )
    return top_k


def handle_prediction(row, allowed_codes):
    if pd.isna(row["ISCO Code"]) or row["ISCO Code"] not in allowed_codes:
        final_prediction = (
            row["alternatives"][0] if row["alternatives"] else row["top-1"]
        )
    else:
        final_prediction = row["ISCO Code"]
    return final_prediction.zfill(4) if final_prediction else None


def check_prediction(row, allowed_codes):
    if (
        is_effectively_null(row["Final Prediction"])
        or row["Final Prediction"] not in allowed_codes
    ):
        print(
            row["id"],
            row["Final Prediction"],
            (
                "IS NA"
                if is_effectively_null(row["Final Prediction"])
                else "NOT IN ALLOWED CODES"
            ),
        )


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
        "-o",
        "--output",
        type=str,
        default="results/",
        help="Path to the output dir",
    )
    return parser.parse_args()


def main():
    # Parse arguments and read configuration
    args = parse_args()
    config_path = args.config
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    cfg = read_config(config_path)

    # Load configurations
    llm_cfg = cfg["llm"]
    dataset_cfg = cfg["dataset"]
    embedding_cfg = cfg["embedding"]
    reranker_cfg = cfg["reranker"]

    # Parse logs and load cleaned data
    df = parse_log_to_dataframe([[llm_cfg["output"]]], [llm_cfg["name"]])

    # Merge dataframes
    merged_df = merge_dataframes([df])

    # Load submission data and merge with parsed data
    submission_df = load_submission_df(dataset_cfg["source"])
    merged_df = merge_with_submission(merged_df, submission_df)

    # Load top-k predictions and labels set
    top_k_preds = load_top_k_predictions(
        reranker_cfg.get("output", embedding_cfg["output"])
    )
    labels_set = load_labels_set(dataset_cfg["labels"])

    # Get allowed codes
    allowed_codes = labels_set["code"].astype(str).tolist()

    # Identify and save missing predictions
    missing_predictions = merged_df[
        merged_df["ISCO Code"].apply(is_effectively_null)
    ]
    print(f"Number of missing predictions: {len(missing_predictions)}")
    missing_ids = missing_predictions["id"].tolist()
    with open(output_dir / "missing_ids.txt", "w") as f:
        f.write("\n".join(missing_ids))

    # Merge top-k predictions
    merge_columns = ["id", "pred_code"]
    if "reranked_preds" in top_k_preds.columns:
        warnings.warn(
            "reranked_preds is in the dataframe, consider using it for top-1 prediction"
        )
        merge_columns.append("reranked_preds")
    else:
        warnings.warn(
            "reranked_preds is not in the dataframe, using top-1 from retrieved predictions"
        )

    merged_df = merged_df.merge(
        top_k_preds[merge_columns],
        left_on="id",
        right_on="id",
        how="right",
    )

    # Handle predictions
    merged_df["top-k"] = merged_df.apply(
        handle_top_k_predictions, axis=1, allowed_codes=allowed_codes
    )
    merged_df["top-1"] = merged_df.apply(get_top_1_predictions, axis=1)
    merged_df["alternatives"] = merged_df.apply(
        handle_alternative_predictions, axis=1, allowed_codes=allowed_codes
    )
    merged_df["Final Prediction"] = merged_df.apply(
        handle_prediction, axis=1, allowed_codes=allowed_codes
    )

    # Check predictions
    merged_df.apply(check_prediction, axis=1, allowed_codes=allowed_codes)

    # Identify and save uncertain predictions
    uncertain_ids = merged_df[
        (merged_df["Final Prediction"] == merged_df["top-1"])
        & (merged_df["ISCO Code"] != merged_df["top-1"])
    ]["id"].tolist()
    with open(output_dir / "uncertain_ids.txt", "w") as f:
        f.write("\n".join(uncertain_ids))

    # Perform fuzzy matching and save results
    result_df, confidence_df, merged_df = fuzzy_matching_df(
        merged_df, llm_cfg["labels"], dataset_cfg["en_output"]
    )
    merged_df.to_csv(output_dir / "merged_predictions.csv", index=False)
    confidence_df.to_csv(output_dir / "confidences.csv", index=False)
    result_df.to_csv(
        output_dir / "classification.csv", index=False, header=False
    )


if __name__ == "__main__":
    main()
