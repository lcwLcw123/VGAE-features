import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGAE_features(nn.Module):
    def __init__(self, adj, dim_in, dim_h, dim_z, gae):
        super(VGAE_features,self).__init__()
        self.dim_z = dim_z
        self.gae = gae
        self.base_gcn = GraphConvSparse(dim_in, dim_h, adj)
        self.gcn_mean = GraphConvSparse(dim_h, dim_z, adj, activation=False)
        self.gcn_logstd = GraphConvSparse(dim_h, dim_z, adj, activation=False)
        self.W = nn.Parameter(torch.rand(dim_z, dim_in))
        #TODO:没有加激活函数activation

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        if self.gae:
            # graph auto-encoder
            return self.mean
        else:
            # variational graph auto-encoder
            self.logstd = self.gcn_logstd(hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
            return sampled_z

    def decode(self, Z):
        Features_pred = Z @ self.W
        return torch.sigmoid(Features_pred)

    def forward(self, X):
        Z = self.encode(X)
        Features_pred = self.decode(Z)
        return Features_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=True):
        super(GraphConvSparse, self).__init__()
        self.weight = self.glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs):
        x = inputs @ self.weight
        x = self.adj @ x
        if self.activation:
            return F.elu(x)
        else:
            return x