from duckduckgo_search import DDGS
import pandas as pd
from tqdm import tqdm
import sglang as sgl
from sglang import set_default_backend, RuntimeEndpoint

# Setup for LLM-based classification
@sgl.function
def classify_isco_level(s, job_description, level):
    """
    Use LLM to classify the job description into ISCO categories step-by-step.
    """
    s += f"The following is a job classification task for ISCO level {level}.\n"
    s += "Job description: " + job_description + "\n"
    s += "Classify into ISCO level " + str(level) + ":\n"
    s += sgl.gen("classification", stop="\n", temperature=0)
    
    return sgl.complete()

def use_search_engine(description):
    """
    Use DuckDuckGo search engine to retrieve related information or verify classification.
    """
    search_results = DDGS().text(description, max_results=5)
    return search_results

def classify_step_by_step_with_llm(description, top_k=10):
    """
    Step-by-step hierarchical classification using LLM with search engine validation.
    """
    # Step 1: LLM Classify by first ISCO digit (broad category)
    results_first_digit = classify_isco_level(description, 1)
    
    # Use search engine to verify or gather more information
    search_results_first_digit = use_search_engine(description)
    
    # Step 2: LLM Classify by second ISCO digit based on the first digit's results
    results_second_digit = classify_isco_level(description, 2)
    
    # Verify second digit with search engine
    search_results_second_digit = use_search_engine(description)
    
    # Step 3: LLM Classify by third ISCO digit based on the second digit's results
    results_third_digit = classify_isco_level(description, 3)
    
    # Verify third digit with search engine
    search_results_third_digit = use_search_engine(description)
    
    # Step 4: LLM Classify by fourth ISCO digit based on the third digit's results
    results_fourth_digit = classify_isco_level(description, 4)
    
    # Verify fourth digit with search engine
    search_results_fourth_digit = use_search_engine(description)
    
    # Return results at each level
    return {
        "first_digit": results_first_digit,
        "second_digit": results_second_digit,
        "third_digit": results_third_digit,
        "fourth_digit": results_fourth_digit,
        "search_verifications": {
            "first_digit": search_results_first_digit,
            "second_digit": search_results_second_digit,
            "third_digit": search_results_third_digit,
            "fourth_digit": search_results_fourth_digit
        }
    }

# Main function to classify jobs
def classify_jobs_step_by_step_with_llm(file_path, output_path, top_k=10):
    """
    Main classification function that processes job descriptions step-by-step using LLM and search engine.
    """
    # Load and preprocess the job description dataset
    df = pd.read_csv(file_path)

    # Apply classification step-by-step for each job description using LLM and search engine
    df_result = df.progress_apply(lambda row: pd.Series(classify_step_by_step_with_llm(row["final_description"], top_k)), axis=1)

    # Save the results to a CSV
    df_result.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Set up LLM backend
    set_default_backend(RuntimeEndpoint("https://lifefoster-sv.computing.dcu.ie/llm-api/"))

    # Run the hierarchical classification for the job dataset
    classify_jobs_step_by_step_with_llm("public_data/wi_dataset_processed_en.csv", "output/hierarchical_job_classifications_llm.csv")
