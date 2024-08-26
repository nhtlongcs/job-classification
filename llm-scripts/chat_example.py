import rich
import openai
client = openai.Client(
    base_url="https://lifefoster-sv.computing.dcu.ie/llm-api/v1", api_key="EMPTY")

# Text completion
response = client.completions.create(
	model="default",
	prompt="The capital of France is",
	temperature=0,
	max_tokens=32,
)
rich.print(response)

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
rich.print(response)
