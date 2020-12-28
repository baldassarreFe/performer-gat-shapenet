import fast_transformers.builders as ftb
import fast_transformers.feature_maps as ftfm
import torch
import torch_geometric as tg
from miniproject.configuration import Architecture


class PerformerModel(torch.nn.Module):
    def __init__(self, conf: Architecture):
        super().__init__()
        dim_model = conf.num_heads * conf.hidden_features
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(conf.in_features, dim_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_model // 2, dim_model),
            torch.nn.LayerNorm(dim_model),
        )
        self.transformer = ftb.TransformerEncoderBuilder.from_kwargs(
            n_layers=conf.num_layers,
            n_heads=conf.num_heads,
            query_dimensions=conf.hidden_features,
            value_dimensions=conf.hidden_features,
            feed_forward_dimensions=2 * conf.num_heads * conf.hidden_features,
            attention_type="linear",
            feature_map=ftfm.Favor.factory(n_dims=256),
        ).get()
        self.classifier = torch.nn.Linear(dim_model, conf.out_features)

    def forward(self, batch: tg.data.Batch) -> torch.Tensor:
        num_graphs = batch.num_graphs
        num_nodes = batch.num_nodes

        # [num_nodes, 6] - [num_nodes, num_features]
        x = torch.cat((batch.x.to(self.device), batch.pos.to(self.device)), dim=1)
        x = self.encoder(x)

        # [num_nodes, num_features] -> [num_graphs, num_nodes, num_features]
        x = x.view(num_graphs, num_nodes // num_graphs, x.shape[1])
        x = self.transformer(x)

        # [num_nodes, num_nodes, num_features] -> [num_graphs, num_features]
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

    @property
    def device(self):
        return self.classifier.weight.device
