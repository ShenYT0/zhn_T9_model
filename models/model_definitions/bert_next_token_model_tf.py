import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class BertNextTokenModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, ff_dim=128, num_layers=2, max_len=32):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embed = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.encoder_layers = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.norm = layers.LayerNormalization()
        self.cls_head = layers.Dense(vocab_size)  # next token prediction

    def call(self, input_ids, training=False):
        seq_len = tf.shape(input_ids)[1]
        pos_ids = tf.range(start=0, limit=seq_len, delta=1)[None, :]
        x = self.embedding(input_ids) + self.position_embed(pos_ids)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        x = self.norm(x)
        last_token = x[:, -1, :]  # only use the last token's hidden state
        return self.cls_head(last_token)  # shape: (batch, vocab_size)
