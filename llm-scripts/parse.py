import re
import json

def parse_final_prediction(response):
    job_id_pattern = r"Job Ad ID:\s*(\d+)"
    prediction_pattern = r'\*\*ISCO Code:\*\*\s*(.*)'
    # prediction_pattern = r'\*\*ISCO Code:\*\*\s*(.*)\n\*\*ISCO Title:\*\*\s*(.*)\n\*\*Confidence:\*\*\s*([\d.]+)'

    # Split the response by the delimiter "-----------------------------------"
    parts = response.split('-----------------------------------')

    # Create a list to hold JSON results
    results = []

    for part in parts:
        # Find all matches in each split part
        matches_job_id = re.findall(job_id_pattern, part)
        matches_prediction = re.findall(prediction_pattern, part)
        if matches_job_id and matches_prediction:
            # Use the first matched Job ID
            job_id = matches_job_id[0]
            for match in matches_prediction:
                # Create a dictionary with the extracted information
                result = {
                    "Job Ad ID": job_id,
                    "ISCO Code": match.strip(),
                    # "ISCO Title": match[1].strip(),
                    # "Confidence": float(match[2])
                }
                # Append to the results list
                results.append(result)

    # Return the results in JSON format
    return json.dumps(results, indent=4)

# Example usage with provided job ad responses
response = """
## Reasoning Process:
... (other parts of the response) ...

**Final Prediction:**

**ISCO Code:** 7132
**ISCO Title:** Spray Painters and Varnishers
**Confidence:** 0.9
**Reasoning:** The job description clearly aligns with the tasks and skills required of a Spray Painter and Varnisher. The emphasis on experience, qualifications, and producing high-quality workmanship in vehicle bodywork strongly supports this classification. 

-----------------------------------

## Reasoning Process:
... (other parts of the response) ...

**Final Prediction:**

**ISCO Code:**  *Not applicable* 
**ISCO Title:** *Not applicable*
**Confidence:** 1.0
**Reasoning:** The job advertisement describes a role that falls outside the scope of the provided ISCO units. The role of a teacher requires a unique set of skills and responsibilities that are not captured in the given ISCO units. 
----------------------------------- 
"""

# Parse the example response
parsed_response = parse_final_prediction(response)

print(parsed_response)
