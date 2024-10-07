from llm.model import GenerativeModelWrapper
import pytest
from dotenv import load_dotenv
import os


@pytest.fixture
def model():
    model_name = "gemini-1.5-flash"
    model_version = "002"
    system_prompt = "As a chief human resources officer, you are tasked with analyzing job advertisements and determining the most appropriate ISCO unit for each job, based on the job description and job title. You have access to a list of potential ISCO units with their descriptions, definitions, and skill types. Analyze the job advertisement and the potential ISCO units, considering the main responsibilities and tasks described in the job ad, the required skills and qualifications, and the level of expertise and autonomy required. Provide a step-by-step reasoning process to determine the most appropriate ISCO unit for this job advertisement. Then, provide your final prediction in the format: ISCO Code: [code] ISCO Title: [title] Confidence: [0-1 scale] Reasoning: [A brief summary of your reasoning]."
    load_dotenv()
    api_key = os.getenv("GENAI_API")
    assert api_key is not None
    model = GenerativeModelWrapper(
        model_name=model_name,
        model_version=model_version,
        system_instruction=system_prompt,
        use_local=False,  # Set to True to use local model
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


def test_generate_content_with_different_prompt(model):
    generation_config = {
        "temperature": 0.25,
    }

    response = model.generate_content(
        "analyze this job ad",
        generation_config=generation_config,
    )
    assert response.text is not None
    assert isinstance(response.text, str)


def test_generate_content_with_high_temperature(model):
    generation_config = {
        "temperature": 0.9,
    }

    response = model.generate_content(
        "hello world",
        generation_config=generation_config,
    )
    assert response.text is not None
    assert isinstance(response.text, str)
