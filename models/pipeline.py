import torch

from custom_tokenizers.jieba_tokenizer import JiebaLikeTokenizer
from models.model_instancies import model1


class Pipeline:
    def __init__(self, model=None, tokenizer=JiebaLikeTokenizer()):
        if model is None:
            model = model1()
            model.load_state_dict(torch.load("../models/model_checkpoints/model1_train2.pth"))
        self.model = model
        self.tokenizer = tokenizer
        self.device = tokenizer.device
        self.model.to(self.device)

    def predict(self, text, topk=5):
        tokenized_dict = self.tokenizer(text, return_tensors=True, add_attention_mask=True)
        input_ids = tokenized_dict["input_ids"].unsqueeze(0)
        attention_mask = tokenized_dict["attention_mask"].unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs["logits"].squeeze(0)
        topk_probs, topk_indices = torch.topk(logits, k=topk)
        decoded_texts = [self.tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in topk_indices]
        return decoded_texts
