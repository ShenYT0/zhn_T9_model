import torch

from custom_tokenizers.jieba_tokenizer import JiebaLikeTokenizer
from models.model_instancies import model1
from tools.t9_hanzi_converter import T9PinyinHanziConverter

import os

import re


class Pipeline:
    def __init__(self, model=None, tokenizer=JiebaLikeTokenizer()):
        if model is None:
            model = model1()
            model_path = os.path.join(os.path.dirname(__file__), "model_checkpoints", "model1_train2.pth")
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model
        self.tokenizer = tokenizer
        self.device = tokenizer.device
        self.model.to(self.device)

    def predict(self, text, topk=500, filter_by_digits=True):
        tokenized_dict = self.tokenizer(text, return_tensors=True, add_attention_mask=True)
        input_ids = tokenized_dict["input_ids"].unsqueeze(0)
        attention_mask = tokenized_dict["attention_mask"].unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs["logits"].squeeze(0)
        topk_probs, topk_indices = torch.topk(logits, k=topk)
        decoded_texts = [self.tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in topk_indices]

        digits = self.get_last_digits(text)
        if filter_by_digits and digits:
            decoded_texts = [decoded_text for decoded_text in decoded_texts
                             if T9PinyinHanziConverter.hanzi2t9(decoded_text).startswith(digits)]

        return decoded_texts

    @staticmethod
    def get_last_digits(text: str) -> str:
        digits_match = re.findall(r'\d+', text)
        last_digits = digits_match[-1] if digits_match else ''
        return last_digits
