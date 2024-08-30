# run.py
import pandas as pd
from tqdm import tqdm
from embedding.utils import preprocess_data, get_query_results, save_results
from embedding.pipeline import RetrieverPipeline

# Preprocess data
df_subset = preprocess_data("public_data/wi_dataset.csv")

# Initialize the query pipeline
model = "BAAI/bge-multilingual-gemma2"
instruction= "Given a job advertisement, retrieve relevant job descriptions that matches the query."
pipeline = RetrieverPipeline("public_data/wi_labels.csv", model=model, instruction=instruction)

# Apply the query results to the dataset
tqdm.pandas()
top_k = 10
df_result = df_subset.progress_apply(lambda row: pd.Series(get_query_results(pipeline, row["description"], top_k=top_k)), axis=1)
df_result.columns = ["pred_code", "pred_label"]

# Concatenate the results with the original subset and save the final dataframe
df_result = pd.concat([df_subset, df_result], axis=1)

# Save the results to a CSV file
save_results(df_result, f"classification_top_{top_k}_{model}.csv")
