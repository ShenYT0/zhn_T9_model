from torch import nn
from transformers import BertModel, BertConfig


class BertNextTokenModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, ff_dim=128, num_layers=2, max_len=32, dropout=0.1):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            intermediate_size=ff_dim,
            num_hidden_layers=num_layers,
            max_position_embeddings=max_len,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        self.bert = BertModel(config)
        self.cls_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs.last_hidden_state
        last_token = last_hidden_state[:, -1, :]
        logits = self.cls_head(last_token)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
