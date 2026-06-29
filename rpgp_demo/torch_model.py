try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None


if nn is not None:

    class RelationClassifier(nn.Module):
        def __init__(self, vocab_size: int, relation_count: int, embedding_dim: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.classifier = nn.Linear(embedding_dim, relation_count)

        def forward(self, input_ids):
            mask = (input_ids != 0).unsqueeze(-1)
            embedded = self.embedding(input_ids)
            summed = (embedded * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths
            return self.classifier(pooled)


    class TextCNNRelationClassifier(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            relation_count: int,
            embedding_dim: int = 64,
            hidden_channels: int = 48,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, hidden_channels, kernel_size)
                for kernel_size in (2, 3, 4)
            ])
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_channels * len(self.convs), relation_count)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                features = torch.relu(conv(embedded))
                pooled.append(torch.max(features, dim=2).values)
            return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


    def build_relation_classifier(
        model_type: str,
        vocab_size: int,
        relation_count: int,
        embedding_dim: int = 64,
        hidden_channels: int = 48,
    ):
        if model_type == "textcnn":
            return TextCNNRelationClassifier(
                vocab_size=vocab_size,
                relation_count=relation_count,
                embedding_dim=embedding_dim,
                hidden_channels=hidden_channels,
            )
        return RelationClassifier(vocab_size, relation_count, embedding_dim)

else:

    class RelationClassifier:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required for RelationClassifier")


    class TextCNNRelationClassifier:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required for TextCNNRelationClassifier")


    def build_relation_classifier(*args, **kwargs):  # pragma: no cover
        raise ModuleNotFoundError("torch is required for build_relation_classifier")
