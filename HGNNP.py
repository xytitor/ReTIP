import dhg
import torch
import torch.nn as nn

class HGNNPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        drop_rate: float = 0.5,
        agg_fun: str='softmax_then_sum',
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.agg_fun=agg_fun

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph,e2v_weight=None) -> torch.Tensor:
        X = self.theta(X)
        X = hg.v2e(X, aggr=self.agg_fun)
        X = hg.e2v(X, aggr=self.agg_fun,e2v_weight=e2v_weight)
        X = self.drop(self.act(X))
        return X
class MultiLayerHGNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, layer_num,agg_fun,drop_rate: float = 0.5):

        super(MultiLayerHGNN, self).__init__()


        dim_list = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]


        self.layers = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            self.layers.append(HGNNPConv(
                in_channels=dim_list[i],
                out_channels=dim_list[i + 1],
                drop_rate=drop_rate,
                agg_fun=agg_fun
            ))

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph, e2v_weight=None) -> torch.Tensor:

        for layer in self.layers:
            X = layer(X, hg, e2v_weight)
        return X