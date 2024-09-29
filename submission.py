from utils import *

txt_logs = ["logs/gemini_log.txt",
            "logs/gemini_log_reversed.txt", "logs/gemini_log_missing.txt"]

merged_df = read_and_parse_log(txt_logs)
merged_df = convert_to_str(merged_df)
merged_df['source'] = 'gemini-1.5-flash'

print(f"DF shape: {merged_df.shape}")
print(f"Number of unique Job Ad IDs: {merged_df['Job Ad ID'].nunique()}")
print(f"Number of unique ISCO Codes: {merged_df['ISCO Code'].nunique()}")

print()
print(
    f"Number of not null ISCO Codes: {len(merged_df[~merged_df['ISCO Code'].apply(is_effectively_null)])}")
print(f"Number of duplicated ids: {merged_df['Job Ad ID'].duplicated().sum()}")

# Add another model
txt_logs = ["logs/gemini-1.5-pro_log.txt",
            "logs/gemini-1.5-pro_log_reversed.txt"]

merged_df_2 = read_and_parse_log(txt_logs)
merged_df_2 = convert_to_str(merged_df_2)
merged_df_2['source'] = 'gemini-1.5-pro'

merged_df = pd.concat([merged_df, merged_df_2])
del merged_df_2

# Add another model
txt_logs = ["logs/gemini-1.5-flash_log.txt",
            "logs/gemini-1.5-flash_log_reversed.txt"]

merged_df_2 = read_and_parse_log(txt_logs)
merged_df_2 = convert_to_str(merged_df_2)
merged_df_2['source'] = 'gemini-1.5-flash-002-higher-level'

merged_df = pd.concat([merged_df, merged_df_2])
del merged_df_2

cleaned_df = pd.read_excel('logs/uncertain_samples_cleaned.xlsx', dtype=str)
cleaned_df = cleaned_df[['id', 'potential']]
cleaned_df.columns = ['Job Ad ID', 'Alternative ISCO Codes']
cleaned_df['ISCO Code'] = None
cleaned_df['Confidence'] = 1.0  # hard-coded confidence score
cleaned_df['source'] = 'cleaned'

cleaned_df['ISCO Code'] = cleaned_df['Alternative ISCO Codes'].apply(
    lambda x: x.split(', ')[0] if isinstance(x, str) else None)

merged_df = pd.concat([merged_df, cleaned_df])
del cleaned_df

merged_df = drop_duplicates_prioritize_non_null_with_confidence(
    merged_df, id_col='Job Ad ID', isco_code_col='ISCO Code', confidence_col='Confidence')

print(f"Number of unique Job Ad IDs: {merged_df['Job Ad ID'].nunique()}")
print(f"Number of unique ISCO Codes: {merged_df['ISCO Code'].nunique()}")
print(
    f"Number of not null ISCO Codes: {len(merged_df[~merged_df['ISCO Code'].apply(is_effectively_null)])}")

submission_df = pd.read_csv(
    'datasets/EU-stats-challenge/submission-phase/wi_dataset.csv')[['id']]
submission_df = convert_to_str(submission_df)
submission_df['id'].nunique()

print(f"Number of unique Job Ad IDs: {submission_df['id'].nunique()}")

merged_df = merged_df.merge(
    submission_df, left_on='Job Ad ID', right_on='id', how='right')

print(f"Number of unique Job Ad IDs: {merged_df['id'].nunique()}")
print(f"Number of not null ISCO Codes: {merged_df['ISCO Code'].notna().sum()}")
# check for both missing predictions and empty predictions
missing_predictions = merged_df[merged_df['ISCO Code'].apply(
    is_effectively_null)]
print(f"Number of missing predictions: {len(missing_predictions)}")
merged_df = merged_df.drop(columns=['Job Ad ID'])

merged_df.sort_values(by='Confidence', ascending=True)

# Matching with labels set and top-k predictions

# top_k_preds = pd.read_csv('top_k_prediction/classification_top_10_stella_en_1.5B_v5.csv')
# top_k_preds = pd.read_csv('top_k_prediction/classification_top_10_stella_en_1.5B_v5_reranked.csv')
# top_k_preds = pd.read_csv('top_k_prediction/classification_top_10_text-embedding-3-large_new.csv', dtype=str)
top_k_preds = pd.read_csv(
    'top_k_prediction/classification_top_10_text-embedding-3-large_new_reranked.csv', dtype=str)
labels_set = pd.read_csv(
    'datasets/EU-stats-challenge/submission-phase/wi_labels.csv', dtype=str)

top_k_preds = convert_to_str(top_k_preds)
labels_set = convert_to_str(labels_set)


# merge merged_df with top_k_preds
if "reranked_preds" in top_k_preds.columns:
    warnings.warn(
        "reranked_preds is in the dataframe, consider using it for top-1 prediction")
    merged_df = merged_df.merge(top_k_preds[[
                                'id', 'pred_code', 'reranked_preds']], left_on='id', right_on='id', how='right')
else:
    warnings.warn(
        "reranked_preds is not in the dataframe, using top-1 from retrieved predictions")
    merged_df = merged_df.merge(
        top_k_preds[['id', 'pred_code']], left_on='id', right_on='id', how='right')

#  Statistics

# number of predictions that are not in the allowed codes
allowed_codes = labels_set['code'].astype(str).tolist()

# check for both missing predictions and empty predictions
missing_predictions = merged_df[merged_df['ISCO Code'].apply(
    is_effectively_null)]
print(f"Number of missing predictions: {len(missing_predictions)}")
missing_ids = missing_predictions['id'].tolist()

with open('missing_ids.txt', 'w') as f:
    f.write('\n'.join(missing_ids))


def handle_top_k_predictions(row):
    top_k = []
    if isinstance(row['pred_code'], str):
        for code in row['pred_code'].split(', '):
            if code in allowed_codes:
                top_k.append(code)
    return top_k

merged_df['top-k'] = merged_df.apply(handle_top_k_predictions, axis=1)


def get_top_1_predictions(row):
    top_1 = row['top-k'][0]
    if "reranked_preds" in row.index:
        if top_1 != row['ISCO Code']:
            # get the first reranked prediction
            top_1 = row['reranked_preds'].split(', ')[0]

    if len(top_1) == 3:
        top_1 = '0' + top_1
    return top_1


merged_df['top-1'] = merged_df.apply(get_top_1_predictions, axis=1)

# handle alternative predictions
def handle_alternative_predictions(row, allowed_codes=allowed_codes):
    top_k = []
    if isinstance(row['Alternative ISCO Codes'], str):
        for code in row['Alternative ISCO Codes'].split(', '):
            if code in allowed_codes:
                top_k.append(code)
    return top_k

merged_df['alternatives'] = merged_df.apply(
    handle_alternative_predictions, axis=1)


not_allowed_predictions = merged_df[~merged_df['ISCO Code'].isin(
    allowed_codes)]
print(
    f"Number of predictions not in the allowed codes: {len(not_allowed_predictions)}")

allowed_predictions = merged_df[merged_df['ISCO Code'].isin(allowed_codes)]
allowed_predictions.sort_values(by='ISCO Code')


# number of allowed predictions that are not top-1
print(
    f"Number of allowed predictions not top-1: {len(allowed_predictions[allowed_predictions['ISCO Code'] != allowed_predictions['top-1']])}")
print(
    f"Top-1 accuracy: {len(allowed_predictions[allowed_predictions['ISCO Code'] == allowed_predictions['top-1']]) / len(allowed_predictions)}")


# number of allowed predictions that are not in top-k
print(
    f"Number of allowed predictions not top-k: {len(allowed_predictions[~allowed_predictions['ISCO Code'].isin(allowed_predictions['top-k'].explode().tolist())])}")
print(
    f"Top-k accuracy: {len(allowed_predictions[allowed_predictions['ISCO Code'].isin(allowed_predictions['top-k'].explode().tolist())]) / len(allowed_predictions)}")


# Handling missing predictions and not allowed predictions
merged_df['Final Prediction'] = None

# Initialize counters
from_isco_code = 0
from_alternatives = 0
from_top_1 = 0
padded = 0


def handle_prediction(row, allowed_codes=allowed_codes):
    global from_isco_code, from_alternatives, from_top_1, padded
    if pd.isna(row['ISCO Code']) or row['ISCO Code'] not in allowed_codes:
        if row['alternatives']:
            final_prediction = row['alternatives'][0]
            from_alternatives += 1
        else:
            final_prediction = row['top-1']
            from_top_1 += 1
    else:
        final_prediction = row['ISCO Code']
        from_isco_code += 1

    if len(final_prediction) < 4:
        num_zeros = 4 - len(final_prediction)
        final_prediction = '0' * num_zeros + final_prediction
        padded += 1
    return final_prediction


merged_df['Final Prediction'] = merged_df.apply(handle_prediction, axis=1)

# Print statistics
total = len(merged_df)
print(f"Total predictions: {total}")
print(f"From ISCO Code: {from_isco_code} ({from_isco_code/total:.2%})")
print(
    f"From alternatives: {from_alternatives} ({from_alternatives/total:.2%})")
print(f"From top-1: {from_top_1} ({from_top_1/total:.2%})")
print(f"Padded predictions: {padded} ({padded/total:.2%})")


def check_prediction(row):
    if is_effectively_null(row['Final Prediction']):
        print(row['id'], row['Final Prediction'], "IS NA")
    else:
        if row['Final Prediction'] not in allowed_codes:
            print(row['id'], row['Final Prediction'], "NOT IN ALLOWED CODES")


merged_df.apply(check_prediction, axis=1).sum()
merged_df['Final Prediction'] = merged_df['Final Prediction'].astype(str)

uncertain_ids = merged_df[(merged_df['Final Prediction'] == merged_df['top-1'])
                          & (merged_df['ISCO Code'] != merged_df['top-1'])]['id'].tolist()
uncertain_ids = [x for x in uncertain_ids if x not in missing_ids]
with open('uncertain_ids.txt', 'w') as f:
    f.write('\n'.join(uncertain_ids))


# Save the merged predictions
merged_df['alternatives'] = merged_df['alternatives'].apply(
    lambda x: ', '.join(x))
merged_df['top-k'] = merged_df['top-k'].apply(lambda x: ', '.join(x))
merged_df.to_csv('merged_predictions.csv', index=False)

merged_df[['id', 'Confidence']].to_csv(
    'confidences.csv', index=False)

merged_df[['id', 'Final Prediction']].to_csv(
    'classification.csv', index=False, header=False)

