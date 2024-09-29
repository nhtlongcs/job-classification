import pandas as pd


def generate_prompt(job_ad, retrieved_job_info_txt, post_prompt=""):
    prompt = f"""
    Given the following job advertisement:
    Title: {job_ad.title}
    Description: {job_ad.final_description}
    If the job ad is not in English, translate it to English.
    
    And the following potential ISCO units with their respective descriptions, definitions, and skill types, sorted by descending relevance:
    {retrieved_job_info_txt}
    
    Analyze the job advertisement and the potential ISCO units. Consider the following:
    1. The job title and description, compare it with provided related job titles.
    2. The main responsibilities and tasks described in the job ad.
    3. The required skills and qualifications.
    4. The major, sub-major, and minor groups and how they relate to the job ad.
    
    Provide a step-by-step reasoning process to determine the most appropriate ISCO unit for this job advertisement.
    If the job advertisement does not match any of the potential ISCO units, provide a rationale for why this is the case, and suggest three (3) alternative ISCO units descending relevance that could potentially match the job advertisement.
    Then, provide your final prediction in the format: 
    ISCO Code (unit): [code]
    ISCO Title: [title]
    Confidence: [0-1 scale]
    Reasoning: [A brief summary of your reasoning]
    Alternative ISCO Codes (unit): [code1, code2, code3].
    Please note that the provided ISCO units are sorted by descending relevance, perform analysis on the provided ISCO units first by the given order, before making any alternative predictions.
    If not confident, provide the higher-level and more certain ISCO unit that the job ad is belong to.
    """
    if post_prompt:
        prompt = f"{prompt}\n{post_prompt}"
    return prompt


def get_retrieved_info(job_ad, k, labels_df: pd.DataFrame, return_text: bool = True):
    if isinstance(job_ad, pd.DataFrame):
        job_ad = job_ad.iloc[0]
    top_k = job_ad.pred_code.split(',')
    top_k = [int(x.strip()) for x in top_k]
    top_k = top_k[:k]
    retrieved_job_info = labels_df[labels_df['code'].isin(top_k)]
    if return_text:
        formatted = ""
        for i, unit in retrieved_job_info.iterrows():
            formatted += f"Detailed description: {unit['description_x']}\n"
            formatted += f"Job definition: {unit['definition']}\n"
            formatted += f"Skill group: {unit['skill_group']}, skill group label: {unit['skill_group_label']}, skill level: {unit['skill_level']}, skill label: {unit['skill_label']}	\n"
            formatted += f"Similar job titles: {unit['english title']}\n"
            formatted += f"Major code: {unit['major']}, major label: {unit['major_label']}\n"
            formatted += f"Sub-major code: {unit['sub_major']}, sub-major label: {unit['sub_major_label']}\n"
            formatted += f"ISCO Code (unit code): {unit['code']}\n"

        return formatted
    else:
        return retrieved_job_info
