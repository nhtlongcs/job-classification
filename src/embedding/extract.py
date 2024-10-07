import numpy as np
import torch
from net import EmbeddingHuggingFaceModel
from tqdm import tqdm


def extract(texts):
    model = EmbeddingHuggingFaceModel("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embs = []
    with torch.inference_mode():
        for text in tqdm(texts):
            input_ids = model.tokenizer(text, return_tensors="pt")["input_ids"]
            attention_mask = model.tokenizer(text, return_tensors="pt")[
                "attention_mask"
            ]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = model(input_ids, attention_mask)
            embs.append(
                output.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            )
    return np.concatenate(embs)


if __name__ == "__main__":
    model = EmbeddingHuggingFaceModel("bert-base-uncased")
    embs = extract(["hello world", "goodbye world"])
    np.save("embs.npy", embs)
