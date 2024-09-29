import pandas as pd
import string
from IPython.utils import io
from langdetect import detect
from polyfuzz import PolyFuzz

abbreviation_dict = {
    'Sr.': 'Senior',
    'Jr.': 'Junior',
    'VP': 'Vice President',
    'Mgr': 'Manager',
    'Dir': 'Director',
    'HR': 'Human Resources',
    'PA': 'Personal Assistant',
    'PR': 'Public Relations',
    'QA': 'Quality Assurance',
}

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # columns = ["id", "title", "final_description"]
    # df_subset = df[columns]
    # df_subset.drop_duplicates(subset="description", inplace=True)
    # df_subset.dropna(inplace=True)
    # df_subset["description"] = df_subset["description"].str.lower()
    # df_subset["description"] = df_subset["description"].str.replace(f'[{string.punctuation}]', '', regex=True)
    return df

def is_english(text):
    try:
        return detect(text) == 'en' 
    except:
        return False

def get_query_results(pipeline, query: str, top_k: int = 5):
    try:
        with io.capture_output() as captured:
            prediction = pipeline.run_query(query, top_k)
        codes = [str(doc.meta['code']) for doc in prediction['retriever']['documents']]
        labels = [str(doc.meta['label']) for doc in prediction['retriever']['documents']]
        return ', '.join(codes), ', '.join(labels)
    except:
        return '', ''

def save_results(df, output_path):
    df.to_csv(output_path, index=False)

def replace_abbreviations(title):
    words = title.split()  # Tokenize the title into words
    transformed_words = [abbreviation_dict.get(word, word) for word in words]  # Replace abbreviations
    return ' '.join(transformed_words)  # Join words back together

def fuzzy_match(from_list, to_list):
    model = PolyFuzz("TF-IDF")
    model.match(from_list, to_list)
    inspect_df = model.get_matches()
    return inspect_df