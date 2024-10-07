import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class EmbeddingHuggingFaceModel(nn.Module):

    def __init__(self, model_name):
        super(EmbeddingHuggingFaceModel, self).__init__()

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, input_ids, attention_mask):
        return self.transformer(input_ids, attention_mask)
