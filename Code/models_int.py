import torch
import torch.nn as nn
from torch_scatter import scatter


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.coe_timediff = 0.2

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        self.tau_emb = nn.Embedding(366, in_dim)
        self.Wtau_attn = nn.Linear(in_dim, attn_dim, bias=False)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, in_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, in_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, in_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, in_dim))

    def forward(self, q_sub, q_rel, q_tau, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, tau, old_idx, new_idx]
        sub = edges[:, 5]
        rel = edges[:, 2]
        obj = edges[:, 6]
        tau = edges[:, 4]  # 边的时间

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        taus_q = q_tau[r_idx]  # 边对应的查询的时间
        tau = torch.where(tau >= 0, tau, taus_q)
        delta_tau = tau - taus_q

        h_tau1 = self.weight_t1 * delta_tau.unsqueeze(1) + self.bias_t1
        h_tau2 = torch.sin(self.weight_t2 * delta_tau.unsqueeze(1) + self.bias_t2)
        h_hau = h_tau1 + h_tau2

        # 以下可以改一下
        inf_time = delta_tau * self.coe_timediff
        # inf_time = torch.exp(inf_time)
        message = hs + hr + h_hau
        alpha = torch.sigmoid(
            self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr) + self.Wtau_attn(h_hau))))

        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new


class TPAR(torch.nn.Module):
    def __init__(self, params, loader):
        super(TPAR, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, taus, mode='train'):
        n = len(subs)

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        q_taus = torch.LongTensor(taus).cuda()

        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        scores_all = []
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)

            hidden = self.gnn_layers[i](q_sub, q_rel, q_taus, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()  # non_visited entities have 0 scores
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all



