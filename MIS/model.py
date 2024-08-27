import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import sys
import networkx as nx
from torch_geometric.utils import from_networkx
from AgentNet.model import AgentNet

class agent_cmp(AgentNet):
    def __init__(self, D):
        super().__init__(num_features=1, hidden_units=128, num_out_classes=1, dropout=0.0, num_steps=16,
                        num_agents=20, reduce="log", node_readout=False, use_step_readout_lin=False,
                        num_pos_attention_heads=1, readout_mlp=False, self_loops=True, post_ln=False,
                        attn_dropout=0.0, no_time_cond=False, mlp_width_mult=2, activation_function="leaky_relu",
                        negative_slope=0.01, input_mlp=True, attn_width_mult=1, importance_init=False,
                        random_agent=False, test_argmax=False, global_agent_pool=True, agent_global_extra=False,
                        basic_global_agent=False, basic_agent=False, bias_attention=True, visited_decay=0.2,
                        sparse_conv=False, mean_pool_only=False, edge_negative_slope=0.2,
                        final_readout_only=False)
        
    def forward(self, A, B, device):
        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        G_A = nx.from_numpy_matrix(A_np)
        G_B = nx.from_numpy_matrix(B_np)
        # 将G_A转化为data.x,data.edge_index,data.edge_attr
        data_A = from_networkx(G_A).to(device)
        data_B = from_networkx(G_B).to(device)
        # 调用AgentNet的forword()方法
        X = torch.cat((torch.ones(data_A.num_nodes, 1), torch.ones(data_B.num_nodes, 1)), dim=0).to(device)
        edge_index = torch.cat((data_A.edge_index, data_B.edge_index + data_A.num_nodes), dim=1).to(device)
        batch = torch.cat((torch.zeros(data_A.num_nodes, dtype=torch.int64), torch.ones(data_B.num_nodes, dtype=torch.int64))).to(device)
        softmax = nn.Softmax()
        out = super().forward(X, edge_index, batch).view(1, 2)
        out = softmax(out)
        return out
    
    def fast_forward(self, x, edge_index, batch):
        out = super().forward(x, edge_index, batch).view(-1)
        index = torch.arange(0, out.size(0)/2, dtype=torch.int64)
        index_0 = index * 2
        index_1 = index_0 + 1
        out = torch.stack([out[index_0], out[index_1]])
        softmax = nn.Softmax()
        out = softmax(out)
        out = out.t()
        return out



class Comparator(nn.Module):

    def __init__(self, D, device, num_gnn_layers=3, num_dense_layers=3):
        super(Comparator, self).__init__()
        assert num_dense_layers > 2  # For skip connection

        self.d = D

        self.gnn_f_layers = nn.ModuleList([
            nn.Linear(3 * D, D) for _ in range(num_gnn_layers)
        ])

        self.gnn_s_layers = nn.ModuleList([
            nn.Linear(3 * D, D) for _ in range(num_gnn_layers)
        ])

        self.gnn_a_layers = nn.ModuleList([
            nn.Linear(3 * D, D) for _ in range(num_gnn_layers)
        ])

        self.gnn_layer_norms = nn.ModuleList([
            nn.LayerNorm(3 * D) for _ in range(num_gnn_layers)
        ])

        self.dense_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.dense_layers.append(nn.Linear(3 * D, D))
            elif i + 1 == num_dense_layers:
                self.dense_layers.append(nn.Linear(2 * D, 1))

            else:
                self.dense_layers.append(nn.Linear(D, D))

        self.dense_layer_norms = nn.ModuleList([
            nn.LayerNorm(D) for _ in range(num_dense_layers - 1)
        ])

        self.to(device)

    # Takes the adjecancy matrx A of the graph and computes the D-dimensional embedding of the graph

    def Embed(self, A, device):
        GELU = nn.GELU()

        # Computes the complement (including self-loops, which we need anyway)
        B = 1 - A

        # Add self loops to original adjacency matrix!!!
        A = A.fill_diagonal_(1)

        n = A.shape[0]
        X = torch.zeros(n, self.d * 3).to(device)

        for i in range(len(self.gnn_a_layers)):
            Y = self.gnn_f_layers[i](X)
            Z = self.gnn_s_layers[i](X)
            W = self.gnn_a_layers[i](X)

            X = GELU(torch.cat((Z, A @ Y, B @ W), dim=-1))
            X = self.gnn_layer_norms[i](X)

        # Global pooling
        X = torch.mean(X, dim=0)

        return X

    def forward(self, A, B, device):
        warnings.filterwarnings("ignore")
        A = A.to(device)  # Add self loops, important!
        B = B.to(device)  # Add self loops, important!
        RELU = nn.ReLU()

        X = self.Embed(A, device)  # compute the embedding of A
        Y = self.Embed(B, device)  # compute the embedding of B

        for i in range(len(self.dense_layers)):
            if i + 1 == len(self.dense_layers):

                X = torch.cat((X, X_0), dim=-1)
                Y = torch.cat((Y, Y_0), dim=-1)

            X = self.dense_layers[i](X)
            Y = self.dense_layers[i](Y)
            if i + 1 < len(self.dense_layers):

                X = self.dense_layer_norms[i](RELU(X))
                Y = self.dense_layer_norms[i](RELU(Y))
            if i == 0:
                X_0, Y_0 = X, Y
        Z = torch.cat((X, Y), dim=0).reshape(1, 2)

        m = torch.nn.Softmax()
        return m(Z)  # Return the input to the FeedForward Net
