import os
import re


def get_last_processed_row(log_file):
    if not os.path.exists(log_file):
        return 0

    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if line.startswith("Job Ad ID:"):
                match = re.search(r"Row: (\d+)", line)
                if match:
                    return int(match.group(1)) + 1
    return 0


def compose_output(job_id, row_idx, isco_code, response):
    output = ""
    output += f"Job Ad ID: {job_id}, Row: {row_idx}, Pred: {isco_code}\n"
    output += "Response:\n"
    output += f"{response}\n\n"
    output += "-----------------------------------\n\n"
    return output
