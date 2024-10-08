import re
import json
import tqdm
import pandas as pd
import numpy as np
import warnings


def switch_api_key(model, API_KEYS, exhausted_key=None):

    if exhausted_key:
        API_KEYS.remove(exhausted_key)
        API_KEYS.append(exhausted_key)

    if not API_KEYS:
        raise ValueError("All API keys have been exhausted")

    new_key = API_KEYS[0]
    model.configure(api_key=new_key)

    print(f"Switched to a new API key: {new_key[:10]}...")

    return new_key, API_KEYS


def parse_final_prediction(response):
    job_id_pattern = r"Job Ad ID:\s*(\d+)"
    prediction_pattern = r"(?:\*{1,2})?ISCO Code \(unit\):\s*(\d+)"
    alternative_prediction_pattern = (
        r"(?:\*{1,2})?Alternative ISCO Codes \(unit\):\s*([\d, ]+)"
    )
    confidence_pattern = r"(?:\*{1,2})?Confidence:\s*(\d+(?:\.\d+)?)"
    parts = response.split("-----------------------------------")

    results = []

    for part in tqdm.tqdm(parts):
        matches_job_id = re.findall(job_id_pattern, part)
        if matches_job_id:
            job_id = matches_job_id[0]
            result = {
                "Job Ad ID": job_id,
            }

            final_prediction_section = (
                part.split("Final Prediction:")[-1]
                if "Final Prediction:" in part
                else part
            )

            matches_prediction = re.findall(
                prediction_pattern, final_prediction_section
            )
            matches_alternative_prediction = re.findall(
                alternative_prediction_pattern, final_prediction_section
            )
            matches_confidence = re.findall(
                confidence_pattern, final_prediction_section
            )

            if matches_prediction:
                isco_code = matches_prediction[0].strip()
                if len(isco_code) == 3:
                    isco_code = "0" + isco_code
                result["ISCO Code"] = isco_code
            if matches_alternative_prediction:
                alt_codes = (
                    matches_alternative_prediction[0].strip().split(", ")
                )
                alt_codes = [
                    "0" + code if len(code) == 3 else code
                    for code in alt_codes
                ]
                result["Alternative ISCO Codes"] = ", ".join(alt_codes)
            if matches_confidence:
                confidence = float(matches_confidence[0])
                result["Confidence"] = confidence

            results.append(result)

    return results


def read_and_parse_log(txt_logs) -> pd.DataFrame:
    all_results = []

    for txt_path in txt_logs:
        with open(txt_path, "r") as f:
            results = parse_final_prediction(f.read())
            all_results.extend(results)
    results_df = pd.DataFrame(all_results)

    return results_df


def convert_to_str(df):
    DTYPE = {
        "id": "str",
        "ISCO Code": "str",
        "Alternative ISCO Codes": "str",
        "pred_code": "str",
    }

    for col in df.columns:
        if col in DTYPE.keys():
            df[col] = df[col].astype(str)
    return df


def drop_duplicates_prioritize_non_null(
    df: pd.DataFrame, id_col="id", isco_code_col="ISCO Code"
) -> pd.DataFrame:
    """
    Drop duplicated rows with the same job ID, prioritizing non-null ISCO Code rows.

    Args:
    df (pd.DataFrame): Input DataFrame with job ID and ISCO Code columns.
    id_col (str): Name of the job ID column.
    isco_code_col (str): Name of the ISCO Code column.

    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """

    df["is_non_null"] = ~df[isco_code_col].apply(is_effectively_null)
    df_sorted = df.sort_values(
        by=[id_col, "is_non_null"], ascending=[True, False]
    )

    print(
        f"Number of duplicated rows: {df_sorted.duplicated(subset=id_col).sum()}"
    )
    print(
        f"Number of duplicated rows with non-null ISCO Code: {df_sorted[df_sorted['is_non_null']][id_col].duplicated().sum()}"
    )

    # Drop duplicates, keeping the first occurrence (which will be non-null if available)
    df_deduplicated = df_sorted.drop_duplicates(subset=id_col, keep="first")
    df_deduplicated = df_deduplicated.drop(
        columns=["is_non_null"]
    ).reset_index(drop=True)

    return df_deduplicated


def drop_duplicates_prioritize_non_null_with_confidence(
    df: pd.DataFrame,
    id_col="id",
    isco_code_col="ISCO Code",
    confidence_col="Confidence",
) -> pd.DataFrame:
    """
    Drop duplicated rows with the same job ID, prioritizing non-null ISCO Code rows and higher confidence scores.

    Args:
    df (pd.DataFrame): Input DataFrame with job ID, ISCO Code, and confidence columns.
    id_col (str): Name of the job ID column.
    isco_code_col (str): Name of the ISCO Code column.
    confidence_col (str): Name of the confidence column.

    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """

    df["is_non_null"] = ~df[isco_code_col].apply(is_effectively_null)
    df[confidence_col] = pd.to_numeric(df[confidence_col], errors="coerce")

    # Sort the dataframe to prioritize non-null rows and higher confidence
    df_sorted = df.sort_values(
        by=[id_col, "is_non_null", confidence_col],
        ascending=[True, False, False],
    )

    print(
        f"Number of duplicated rows: {df_sorted.duplicated(subset=id_col).sum()}"
    )
    print(
        f"Number of duplicated rows with non-null ISCO Code: {df_sorted[df_sorted['is_non_null']][id_col].duplicated().sum()}"
    )

    # Drop duplicates, keeping the first occurrence (which will be non-null and highest confidence if available)
    df_deduplicated = df_sorted.drop_duplicates(subset=id_col, keep="first")
    df_deduplicated = df_deduplicated.drop(
        columns=["is_non_null"]
    ).reset_index(drop=True)

    return df_deduplicated


def is_effectively_null(value):
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in ["", "nan", "none", "null"]
    return False
