import jieba
import torch

from pathlib import Path


jieba.load_userdict("../data/training_data/jieba_dict.txt")
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))


class JiebaLikeTokenizer:
    def __init__(self, vocab_path=None, unk_token="[UNK]", pad_token="[PAD]", device=device):
        if vocab_path is None:
            vocab_path = Path(__file__).parent / "jieba_vocab.txt"
        else:
            vocab_path = Path(vocab_path)

        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = f.readlines()
            vocab.append(unk_token)
            vocab.append(pad_token)

        self.token_to_id = {word.strip() : i for i, word in enumerate(vocab)}
        self.id_to_token = {i : word.strip() for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.device = device

        jieba.initialize()

    def tokenize(self, text, padding=True, padding_length=32, truncation=True, max_len=32):
        tokens = jieba.lcut(text)
        result = []
        for token in tokens:
            if not token.strip():
                continue
            if token.isdigit():
                result.extend(list(token))
            else:
                result.append(token)

        if truncation and len(result) > max_len:
            result = result[:padding_length]
        if padding and len(result) < padding_length:
            result.extend([self.pad_token] * (padding_length - len(result)))

        return result

    def convert_tokens_to_ids(self, tokens, padding=True, padding_length=32):
        if padding:
            if len(tokens) < padding_length:
                [tokens.append(self.pad_token) for _ in range(padding_length - len(tokens))]
        return [self.token_to_id.get(tok, self.unk_token_id) for tok in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def add_attention_mask(self, input_ids):
        return [0 if input_id == self.pad_token_id else 1 for input_id in input_ids]

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in {self.unk_token, self.pad_token}]
        return "".join(tokens)

    def __call__(self, text, return_tensors=False, add_attention_mask=False):
        tokenized_dict = dict()
        tokenized_dict["tokens"] = self.tokenize(text)
        tokenized_dict["input_ids"] = self.convert_tokens_to_ids(tokenized_dict["tokens"])
        if add_attention_mask:
            tokenized_dict["attention_mask"] = torch.tensor(self.add_attention_mask(tokenized_dict["input_ids"])).to(self.device)
        if return_tensors:
            tokenized_dict["input_ids"] = torch.tensor(tokenized_dict["input_ids"]).to(self.device)
        return tokenized_dict

    def get_vocab_size(self):
        return self.vocab_size
