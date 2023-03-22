
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchdiffeq import odeint
from tools.options import Options
opt = Options().parse()





class GraphAttentionLayer(nn.Module):
    def __init__(self, 
        in_features, 
        hidden_features,
        concat=False,
        n_heads=4):
        super(GraphAttentionLayer, self).__init__()
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

        
        self.adj = torch.ones([opt.subseq_length, opt.subseq_length]).float().cuda()


        if opt.gattnorm=='ln':
            self.norm = nn.LayerNorm(in_features)
        if opt.gattactivation=='gelu':
            self.activation = nn.GELU()
        if opt.gattactivation=='relu':
            self.activation = nn.ReLU(True)




    def forward(self, t, h):

        b, subseq_length, c = h.shape
        h_org = h



        # ---- w first
        out = []
        for i in range(self.n_heads):
            h = h_org
            W = self.W_list[i]

            h = W(h)

            identity = h
            # h = F.softmax(h, dim=-1)
            attention = h @ h.transpose(-2,-1) # [1608,1608]
            attention = attention * self.adj
            # attention = adj
            attention = F.softmax(attention, -1)
            h = attention @ identity

            out.append(h)

        out = torch.cat(out, dim=-1)



        if opt.gattnorm is not None:
            out = self.norm(out)

        if opt.gattactivation is not None:
            out = self.activation(out)

     

        return out




class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()


        modules_list = []
        for i in range(opt.odefc):
            modules_list.append(nn.Linear(dim, dim))
            if opt.gattnorm=='ln':
                modules_list.append(nn.LayerNorm(dim))
            if opt.gattactivation=='gelu':
                modules_list.append(nn.GELU())
            if opt.gattactivation=='relu':
                modules_list.append(nn.ReLU(True))
        
        
        self.seq = nn.Sequential(
            *modules_list
        )

        a=1


    def forward(self, t, x): 
        x = self.seq(x)
        return x




class Conv1dSeq(nn.Module):
    def __init__(self, dim):
        super(Conv1dSeq, self).__init__()
        self.dim = dim

        self.conv1d = nn.Conv1d(dim,dim,opt.subseq_length,1,padding=opt.subseq_length//2)
        self.norm = nn.BatchNorm1d(dim)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        b,subseq,c = x.shape
        assert c==self.dim
        x = x.permute(0,2,1) # [b,c,subseq]
        x = self.conv1d(x)
        x = self.norm(x)
        x = self.activation(x)

        x = x.permute(0,2,1) # [b,subseq,c]

        return x


        


class GATT(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 n_heads,
                 activation=F.leaky_relu,
                 dropout=0.0,):

        """Dense version of GAT."""
        super(GATT, self).__init__()

        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.n_heads = n_heads
        self.activation = activation


        


        self.adj = torch.ones([opt.subseq_length, opt.subseq_length]).float().cuda()

        
        self.norm_list = nn.ModuleList([
            nn.LayerNorm(self.in_features),
            nn.LayerNorm(self.hidden_features*self.n_heads),
            nn.LayerNorm(self.hidden_features*self.n_heads),
        ])



        self.pde_layers = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_features, n_heads=self.n_heads),
            GraphAttentionLayer(hidden_features * n_heads, hidden_features, n_heads=self.n_heads),
            GraphAttentionLayer(hidden_features * n_heads, hidden_features, n_heads=self.n_heads),
        ])

        self.ode_layers = nn.ModuleList([
            ODEFunc(dim=hidden_features * n_heads),
            ODEFunc(dim=hidden_features * n_heads),
            ODEFunc(dim=hidden_features * n_heads),
        ])        





 


        self.integration_time = torch.tensor([0, 1]).float().cuda()




    def forward(self, x):
        b, subseq_length, c = x.shape


        identity_gatt = x
        # if opt.sumout:
        #     output_list = []

        for i, pde_layer in enumerate(self.pde_layers):


            norm = self.norm_list[i]
            ode_layer = self.ode_layers[i]


            x = norm(x)



            x = odeint(func=pde_layer, y0=x, t=self.integration_time, method='euler')[-1]
            x = odeint(func=ode_layer, y0=x, t=self.integration_time, method='euler')[-1]






        

        return x

