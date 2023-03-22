import torch
import torch.nn as nn
from tools.options import Options
from tools.utils import set_seed
import torch.nn.functional as F
from torchdiffeq import odeint
set_seed(7)
opt = Options().parse()



class VVITLayer(nn.Module):
    def __init__(self, 
        in_features, 
        hidden_features,
        n_heads,
        adj_h):
        super(VVITLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.n_heads = n_heads


        self.W_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()

        for i_head in range(self.n_heads):
            gain = nn.init.calculate_gain('leaky_relu')

            W = nn.Linear(in_features, hidden_features)
            nn.init.xavier_uniform_(W.weight.data, gain=gain)

            self.W_list.append(W)



        self.W = nn.Linear(in_features, hidden_features)
        
        self.adj = torch.ones([adj_h, adj_h], requires_grad=False).float().cuda()

        self.norm = nn.LayerNorm(in_features)
        self.activation = nn.ReLU(True)

        self.integration_time = torch.tensor([0, 1], requires_grad=False).float().cuda()



        self.fcseq_norm = nn.LayerNorm(in_features)
        fcseq_hidden_features = 256
        self.fcseq = nn.Sequential(
            nn.Linear(in_features, fcseq_hidden_features),
            nn.ReLU(),
            nn.Linear(fcseq_hidden_features, in_features),
            nn.ReLU(),
        )



    def forward_attention(self, t, x):
        x_org = x
        # ---- w first
        out = []
        for i in range(self.n_heads):
            x = x_org
            W = self.W_list[i]

            x = W(x)

            identity = x
            attention = x @ x.transpose(-2,-1) # [1608,1608]
            attention = attention * self.adj
            attention = F.softmax(attention, -1)
            x = attention @ identity

            out.append(x)

        out = torch.cat(out, dim=-1) # [64,80,512]
        out = self.activation(out)

        return out


    def forward_fcseq(self, t, x):
        out = self.fcseq(x)

        return out





    def forward(self, x):

        b, c, h, w = x.shape
        x = x.view(-1, opt.subseq_length, c, h, w)
        x = x.view(-1, opt.subseq_length, c, h*w)
        x = x.permute(0,1,3,2) # [b, subseq_length, hw, c]
        x = x.contiguous()
        x = x.view(-1, opt.subseq_length*h*w, c)


        
        x = self.norm(x)



        out = odeint(func=self.forward_attention, y0=x, t=self.integration_time,method='euler')[-1]


        out = self.fcseq_norm(out)
        out = odeint(func=self.forward_fcseq, y0=out, t=self.integration_time,method='euler')[-1]


        out = out.view(-1, opt.subseq_length, h*w, c) # [b, subseq_length, hw, c]
        out = out.permute(0,1,3,2 ) # [b, subseq_length, c, hw]
        out = out.view(-1, opt.subseq_length, c, h, w) # [b, subseq_length, c, h, w]
        out = out.view(-1, c, h, w) # [b*subseq_length, c, h, w]


        

        return out
