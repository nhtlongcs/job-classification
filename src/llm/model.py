import google.generativeai as genai
import openai
import sglang as sgl
from sglang import  set_default_backend, RuntimeEndpoint


class GenerativeModelWrapper:
    def __init__(
        self,
        model_name,
        model_version,
        system_instruction,
        use_local=False,
        host=None,
    ):
        self.use_local = use_local
        if self.use_local:
            self.model = self.load_local_model(f'{host}/v1')
            set_default_backend(RuntimeEndpoint(host))
        else:
            self.model = genai.GenerativeModel(
                f"{model_name}-{model_version}",
                system_instruction=system_instruction,
            )

    def load_local_model(self, host):
        return openai.Client(base_url=host, api_key="EMPTY")

    def configure(self, api_key):
        if not self.use_local:
            genai.configure(api_key=api_key)

    def generate_content(self, prompt, generation_config):
        if self.use_local:
            # prompt = prompt + " \output:"
            return self._generate_content_local(prompt, generation_config)
        else:
            return self.model.generate_content(
                prompt, generation_config=generation_config
            )
    def generate_batch_content(self, prompts, generation_config):
        if self.use_local:
            return self._generate_batch_content_local(prompts, generation_config)
        else:
            raise f"Batch generation not supported for {self.model_name}"
    
    def _generate_batch_content_local(self, prompts, generation_config):
        @sgl.function
        def sgl_output(s, prompt):
            s += prompt + " \\output: ## ISCO Unit Analysis for " + sgl.gen("output", temperature=generation_config.get("temperature", 0), max_tokens=16000)
        states = sgl_output.run_batch(
            [
                {"prompt": p} for p in prompts
            ],
            progress_bar=True,
        )
        return [state['output'] if 'output' in state else '' for state in states ]
    
    def _generate_content_local(self, prompt, generation_config):
        # Text completion
        response = self.model.completions.create(
            model="default",
            prompt=prompt + " \\output: ## ISCO Unit Analysis for ",
            temperature=generation_config.get("temperature", 0),
            max_tokens=16000,
        )
        response = response.choices[0]
        # response = self.model.chat.completions.create(
        #     model="default",
        #     messages=[
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=generation_config.get("temperature", 0),
        #     max_tokens=8200,
        # )
        # response.choices[0].message.content
        return response
