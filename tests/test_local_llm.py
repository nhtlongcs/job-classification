from llm.model import GenerativeModelWrapper
import pytest
from dotenv import load_dotenv
import openai
import os
import rich


def test_simple():

    client = openai.Client(base_url="http://0.0.0.0:30000/v1", api_key="EMPTY")

    # Text completion
    _ = client.completions.create(
        model="default",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=32,
    )

    # Chat completion
    _ = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "List 3 countries and their capitals.",
            },
        ],
        temperature=0,
        max_tokens=64,
    )


@pytest.fixture
def model():
    model_name = None
    model_version = None
    api_key = None
    system_prompt = "As a chief human resources officer, you are tasked with analyzing job advertisements and determining the most appropriate ISCO unit for each job, based on the job description and job title. You have access to a list of potential ISCO units with their descriptions, definitions, and skill types. Analyze the job advertisement and the potential ISCO units, considering the main responsibilities and tasks described in the job ad, the required skills and qualifications, and the level of expertise and autonomy required. Provide a step-by-step reasoning process to determine the most appropriate ISCO unit for this job advertisement. Then, provide your final prediction in the format: ISCO Code: [code] ISCO Title: [title] Confidence: [0-1 scale] Reasoning: [A brief summary of your reasoning]."
    load_dotenv()
    genai_url = os.getenv("GENAI_URL")

    model = GenerativeModelWrapper(
        model_name=model_name,
        model_version=model_version,
        system_instruction=system_prompt,
        use_local=True,  # Set to True to use local model
        host=genai_url,
    )
    model.configure(api_key)
    return model


def test_generate_content(model):
    generation_config = {
        "temperature": 0.25,
    }

    response = model.generate_content(
        "hello world",
        generation_config=generation_config,
    )
    assert response.text is not None
    assert isinstance(response.text, str)
