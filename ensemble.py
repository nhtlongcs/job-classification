from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from pathlib import Path
def merge_predictions(codes):
    """
    Create a final ISCO code prediction by taking all the predictions codes.
    
    input: 
        [(code, score), (code, score), ...]
    output:
        final_code
    
    """
    # Prepare code dictionary
    code_dict = defaultdict(float)

    for code, score in codes:
        assert len(str(code)) == 4, f"Code {code} is not 4 digits"

    for code, score in codes:
        code_dict[str(code)] += score  # Merge scores if same code appears multiple times

    final_code = ""
    prefix_dict = code_dict

    # Process each digit position
    for d in range(4):
        # Voting mechanism to determine the most likely digit at position 'd'
        vote = defaultdict(float)
        for code, score in prefix_dict.items():
            vote[code[d]] += score

        # Choose the digit with the maximum score
        final_digit = max(vote, key=vote.get)
        final_code += final_digit

        # Filter codes that match the current prefix
        prefix_dict = {code: score for code, score in prefix_dict.items() if code.startswith(final_code)}
    
    return final_code



def test():
    print(merge_predictions([(1111, 0.5), (1123, 0.55), (1313, 0.7)]))
    print(merge_predictions([(1111, 0.5), (1112, 0.5), (1113, 0.7)]))
    print(merge_predictions([(1111, 0.5), (1112, 0.5)]))


ensemble_files = [
    "top_k_prediction/merged_predictions_latest.csv",
    # "top_k_prediction/merged_predictions_2024-09-26.csv",
    # "top_k_prediction/merged_predictions_2024-09-22.csv",
    "top_k_prediction/fuzzy_match_merged_predictions_2024-09-23.csv",
]
factors = [1, 1]
    
compare_file = "top_k_prediction/merged_predictions_latest.csv"

def main():
    
    dfs = []

    for file in ensemble_files:
        df = pd.read_csv(file)
        dfs.append(df)
    prediction_col = "ISCO Code"
    prediction_col = "Final Prediction"
    final_df = {"id": [], prediction_col: []}

    for filename in ensemble_files:
        final_df[Path(filename).stem] = []
    uids = set(dfs[0]["id"].unique())
    for i, df in enumerate(dfs):
        dfs[i] = dfs[i].set_index("id")
        dfs[i][prediction_col] = dfs[i][prediction_col].fillna(0)
        dfs[i][prediction_col] = dfs[i][prediction_col].astype(int).astype(str).apply(lambda x: x.zfill(4))
        assert len(uids.intersection(set(df["id"].unique()))) == len(uids), f"Dataframe {i} should have the same ids as the first dataframe"

    for uid in tqdm(uids):
        codes = []
        final_df["id"].append(uid)
        for i, df in enumerate(dfs):
            if "Confidence" not in df.columns:
                df["Confidence"] = 0.7
            code = df.loc[uid][prediction_col]
            if code == "0000":
                score = 0
            else:
                score = df.loc[uid]["Confidence"] if df.loc[uid]["Confidence"] else 0.7
            score = score * factors[i]
            codes.append((code, score))
            final_df[Path(ensemble_files[i]).stem].append(df.loc[uid][prediction_col])
        
        final_code = merge_predictions(codes)
        final_df[prediction_col].append(final_code)

    final_df = pd.DataFrame(final_df)
    final_df.to_csv("ensemble_predictions.csv", index=False)
    error_codes = len(final_df[final_df[prediction_col] == "0000"])
    print(f"There are {error_codes} errors in the final predictions")

    # Compare with the latest predictions
    compare_df = pd.read_csv(compare_file)
    compare_df = compare_df.set_index("id")
    compare_df[prediction_col] = compare_df[prediction_col].fillna(0).astype(int).astype(str).apply(lambda x: x.zfill(4))
    final_df = final_df.set_index("id")
    count = 0 
    for uid in uids:
        if final_df.loc[uid][prediction_col] != compare_df.loc[uid][prediction_col]:
            print(f"{uid}: {compare_df.loc[uid][prediction_col]} -> {final_df.loc[uid][prediction_col]}")
            count += 1
    print(f"Total {count} predictions are different")

if __name__ == "__main__":
    # test()
    main()