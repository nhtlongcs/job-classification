import google.generativeai as genai
import openai


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
            self.model = self.load_local_model(host)
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
