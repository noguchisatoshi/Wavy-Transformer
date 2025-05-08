# models/embeddings.py
import torch
import torch.nn as nn

class CustomBertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, max_position_embeddings=512, dropout_prob=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        # epsを1e-6に変更して数値安定性を向上
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 平均0, 標準偏差0.02で初期化
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        tok_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + pos_embeds + tok_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
