try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class JointExtractionModel(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            relation_count: int,
            embedding_dim: int = 96,
            hidden_size: int = 64,
            max_length: int = 64,
        ):
            super().__init__()
            self.relation_count = relation_count
            self.max_length = max_length
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.encoder = nn.LSTM(
                embedding_dim,
                hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            encoded_size = hidden_size * 2
            self.relation_head = nn.Linear(encoded_size, relation_count)
            self.subject_start = nn.Linear(encoded_size, relation_count)
            self.subject_end = nn.Linear(encoded_size, relation_count)
            self.object_start = nn.Linear(encoded_size, relation_count)
            self.object_end = nn.Linear(encoded_size, relation_count)
            self.subject_bias = nn.Parameter(torch.zeros(relation_count, max_length, max_length))
            self.object_bias = nn.Parameter(torch.zeros(relation_count, max_length, max_length))

        def forward(self, input_ids):
            mask = (input_ids != 0).unsqueeze(-1)
            embedded = self.embedding(input_ids)
            encoded, _ = self.encoder(embedded)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = (encoded * mask).sum(dim=1) / lengths
            relation_logits = self.relation_head(pooled)
            subject_logits = self._span_logits(encoded, self.subject_start, self.subject_end, self.subject_bias)
            object_logits = self._span_logits(encoded, self.object_start, self.object_end, self.object_bias)
            return {
                "relation_logits": relation_logits,
                "subject_logits": subject_logits,
                "object_logits": object_logits,
            }

        def _span_logits(self, encoded, start_layer, end_layer, bias):
            start_logits = start_layer(encoded).permute(0, 2, 1).unsqueeze(-1)
            end_logits = end_layer(encoded).permute(0, 2, 1).unsqueeze(-2)
            seq_len = encoded.shape[1]
            return start_logits + end_logits + bias[:, :seq_len, :seq_len].unsqueeze(0)

else:

    class JointExtractionModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required for JointExtractionModel")
