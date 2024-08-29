import sglang as sgl
from sglang import  set_default_backend, RuntimeEndpoint


@sgl.function
def few_shot_qa(s, question):
    s += """The following are questions with answers.
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
"""
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)

if __name__ == "__main__":
    set_default_backend(RuntimeEndpoint("https://lifefoster-sv.computing.dcu.ie/llm-api/"))

    states = few_shot_qa.run_batch(
        [
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of the Vietnam?"},
            {"question": "What is the capital of China?"},
        ]
    )

    for s in states:
        print(s["answer"])
