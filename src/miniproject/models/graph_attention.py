import torch
import torch_geometric as tg
from miniproject.configuration import Architecture

class GraphAttentionModel(torch.nn.Module):
    def __init__(self, conf: Architecture):
        super().__init__()
        dim_model = conf.num_heads * conf.hidden_features
        self.knearest=tg.transforms.KNNGraph(conf.knearest)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(conf.in_features, dim_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_model // 2, dim_model),
            torch.nn.LayerNorm(dim_model),
        )
        self.gat = GatEncoder(
            num_layers=conf.num_layers,
            features=conf.hidden_features,
            num_heads=conf.num_heads,
        )
        self.classifier = torch.nn.Linear(dim_model, conf.out_features)

    def forward(self, batch: tg.data.Batch) -> torch.Tensor:
        num_graphs = batch.num_graphs
        num_nodes = batch.num_nodes

        # [num_nodes, 6] - [num_nodes, num_features]
        x = torch.cat((batch.x.to(self.device), batch.pos.to(self.device)), dim=1)
        x = self.encoder(x)


        edge_index = self.knearest(batch).edge_index.to(self.device)
        x = self.gat(x, edge_index)

        # [num_nodes, num_features] -> [num_graphs, num_features]
        x = tg.nn.global_mean_pool(x, batch.batch.to(self.device), num_graphs)
        logits = self.classifier(x)
        return logits

    @property
    def device(self):
        return self.classifier.weight.device


class GatBlock(torch.nn.Module):
    def __init__(self, features: int, num_heads: int, dropout: float = 0.1):
        """Graph Attention Block

        Args:
            features (int): features per-head
            num_heads (int): [description]
            dropout (float, optional): [description]. Defaults to 0.1.
        """
        super().__init__()
        dim_model = features * num_heads
        self.gat = tg.nn.GATConv(
            in_channels=dim_model,
            out_channels=features,
            heads=num_heads,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(dim_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim_model, 2 * dim_model),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * dim_model, dim_model),
            torch.nn.Dropout(dropout),
        )
        self.norm2 = torch.nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u = self.norm1(x + self.dropout(self.gat(x, edge_index)))
        z = self.norm2(u + self.ff(u))
        return z


class GatEncoder(torch.nn.Module):
    def __init__(
        self, num_layers: int, features: int, num_heads: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            GatBlock(features, num_heads, dropout) for i in range(num_layers)
        ])
        dim_model = features * num_heads
        self.norm = torch.nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x, edge_index)
        return self.norm(x)
