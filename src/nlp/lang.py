from langdetect import detect as detect_language

import pandas as pd
import string
from IPython.utils import io
from langdetect import detect
from polyfuzz import PolyFuzz
import translators as ts
from tqdm import tqdm

tqdm.pandas()

abbreviation_dict = {
    "Sr.": "Senior",
    "Jr.": "Junior",
    "VP": "Vice President",
    "Mgr": "Manager",
    "Dir": "Director",
    "HR": "Human Resources",
    "PA": "Personal Assistant",
    "PR": "Public Relations",
    "QA": "Quality Assurance",
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
        return detect(text) == "en"
    except Exception as e:
        print(f"Error detecting language: {e}")
        return False


def get_query_results(pipeline, query: str, top_k: int = 5):
    try:
        with io.capture_output() as _:
            prediction = pipeline.run_query(query, top_k)
        codes = [
            str(doc.meta["code"])
            for doc in prediction["retriever"]["documents"]
        ]
        labels = [
            str(doc.meta["label"])
            for doc in prediction["retriever"]["documents"]
        ]
        return ", ".join(codes), ", ".join(labels)
    except Exception as e:
        print(f"Error getting query results: {e}")
        return "", ""


def save_results(df, output_path):
    df.to_csv(output_path, index=False)


def replace_abbreviations(title):
    words = title.split()  # Tokenize the title into words
    transformed_words = [
        abbreviation_dict.get(word, word) for word in words
    ]  # Replace abbreviations
    return " ".join(transformed_words)  # Join words back together


def fuzzy_match(from_list, to_list):
    model = PolyFuzz("TF-IDF")
    model.match(from_list, to_list)
    inspect_df = model.get_matches()
    return inspect_df


def split_translate_merge(sentence, lang, verbose=False):
    if lang == "en" or lang is None:
        return sentence
    try:
        chunk_size = 5000
        chunks = [
            sentence[i : i + chunk_size]
            for i in range(0, len(sentence), chunk_size)
        ]
        translated_chunks = [
            ts.translate_text(
                chunk,
                from_language=lang,
                to_language="en",
                translator="google",
            )
            for chunk in chunks
        ]
        merged_sentence = "".join(translated_chunks)
        if verbose:
            print(merged_sentence)
        return merged_sentence
    except Exception as e:
        if verbose:
            print(f"Translation error: {e}")
        return ""


def translate_process_data(df_dataset, verbose=False):
    df_dataset["lang"] = df_dataset.progress_apply(
        lambda x: (
            detect(x["description"])
            if (
                isinstance(x["description"], str) and len(x["description"]) > 5
            )
            else detect(x["title"])
        ),
        axis=1,
    )

    df_dataset["translated_description"] = df_dataset.progress_apply(
        lambda x: split_translate_merge(x["description"], x["lang"], verbose),
        axis=1,
    )

    df_dataset["translated_title"] = df_dataset.progress_apply(
        lambda x: split_translate_merge(x["title"], x["lang"], verbose),
        axis=1,
    )
    unwanted_keywords = [
        "You will now be redirected",
        "reCAPTCHA check page reCAPTCHA check page",
        "We use cookies to ensure",
        "Your browser does not support",
        "Vacancies & jobs in the Joblift job search We have all the jobs. Therefore, you only need one page for your job search: You will now be redirected to our partner. If you are not redirected within the next few seconds, please click here",
        "Job - Job ads - Job offers - CV Market To continue browsing our portal, contact +370 5 219 0032",
        "Home Job offer Are you looking for people? Login Register Service price list © 2021 Zoznam, s.r.o. All rights reserved.",
        "Send CV In the next step, you will be able to choose",
    ]
    df_dataset["description"] = df_dataset["description"].fillna("")
    df_dataset["translated_description"] = df_dataset[
        "translated_description"
    ].fillna("")
    df_dataset["final_description"] = df_dataset.progress_apply(
        lambda x: (
            x["translated_title"]
            if any(
                keyword in x["translated_description"]
                for keyword in unwanted_keywords
            )
            or len(x["translated_description"]) < 50
            else x["translated_title"] + " " + x["translated_description"]
        ),
        axis=1,
    )
    return df_dataset
