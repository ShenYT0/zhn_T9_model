from model_definitions.bert_next_token_model import BertNextTokenModel
from custom_tokenizers.jieba_tokenizer import JiebaLikeTokenizer


tokenizer = JiebaLikeTokenizer()


def model1():
    """Best model found by ture.ipynb"""
    return BertNextTokenModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=256,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
    )


def model2():
    """A larger model since the last one was underfitting"""
    return BertNextTokenModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=512,
        num_heads=8,
        ff_dim=1024,
        num_layers=4,
    )