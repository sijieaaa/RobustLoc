


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from network.gatt import GATT
from network.vvit import VVITLayer
from tools.utils import set_seed
set_seed(7)
from tools.options import Options
opt = Options().parse()





class RobustLoc(nn.Module):
    def __init__(self, feature_extractor):
        super(RobustLoc, self).__init__()


        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features


      

        fe_tq_planes_l4 = 512
        self.fc_xyzwpq_l4 = nn.ModuleList([
            nn.Linear(fe_tq_planes_l4, 1) for _ in range(6)
        ])
        fe_tq_planes_l3 = 256
        self.fc_xyzwpq_l3 = nn.ModuleList([
            nn.Linear(fe_tq_planes_l3, 1) for _ in range(6)
        ])
        fe_tq_planes_l2 = 128
        self.fc_xyzwpq_l2 = nn.ModuleList([
            nn.Linear(fe_tq_planes_l2, 1) for _ in range(6)
        ])
        fe_tq_planes_l1 = 64
        self.fc_xyzwpq_l1 = nn.ModuleList([
            nn.Linear(fe_tq_planes_l1, 1) for _ in range(6)
        ])


        fe_tq_planes_final = (opt.subseq_length+2) * fe_out_planes
        self.fc_xyzwpq_final = nn.ModuleList([
            nn.Linear(fe_tq_planes_final, 1) for _ in range(6)
        ])



        self.gatt = GATT(512, out_features=None, hidden_features=64, n_layers=3, n_heads=8)
        
        tq_features = 128
        self.fcseq_t = nn.Sequential(
            nn.Linear(512,tq_features),
            nn.ReLU(True),
        )
        self.fcseq_q = nn.Sequential(
            nn.Linear(512,tq_features),
            nn.ReLU(True),
        )



        self.lastfc_t = nn.ModuleList(
            nn.Linear(tq_features,1) for _ in range(3)
        )
        self.lastfc_q = nn.ModuleList(
            nn.Linear(tq_features,1) for _ in range(3)
        )

        self.vvit_l40 = VVITLayer(in_features=512, hidden_features=64, n_heads=8, adj_h=opt.subseq_length*(4**2))



    def avgpool_flatten(self,x):
        x = self.feature_extractor.avgpool(x) # [64,c,1,1]
        x = torch.flatten(x, 1) # [64,c]
        return x



    def forward_backbone(self,x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)


        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)


        x_layer1 = x # [64,32,32]


        x = self.feature_extractor.layer2(x)


        x_layer2 = x # [128,16,16]


        x = self.feature_extractor.layer3(x)


        x_layer3 = x # [256,8,8]


        x = self.feature_extractor.layer4(x)

        x = self.vvit_l40(x)

        x_layer4 = x # [512,4,4]


        return x_layer1, x_layer2, x_layer3, x_layer4





    def forward(self, x, timestamps):
        b,subseq_length,c,h,w = x.shape


        x = x.view(-1,c,h,w)

        x_layer1, x_layer2, x_layer3, x_layer4 = self.forward_backbone(x)



        x = x_layer4


        


        x = F.relu(x)


        x = self.avgpool_flatten(x)






        xyzwpq_l4 = torch.cat([fc(x) for fc in self.fc_xyzwpq_l4], dim=1)

        x_layer3 = self.avgpool_flatten(x_layer3)
        xyzwpq_l3 = torch.cat([fc(x_layer3) for fc in self.fc_xyzwpq_l3], dim=1)

        x_layer2 = self.avgpool_flatten(x_layer2)
        xyzwpq_l2 = torch.cat([fc(x_layer2) for fc in self.fc_xyzwpq_l2], dim=1)

        x_layer1 = self.avgpool_flatten(x_layer1)
        xyzwpq_l1 = torch.cat([fc(x_layer1) for fc in self.fc_xyzwpq_l1], dim=1)



        x = x.view(b, subseq_length, 512)
        x = self.gatt(x).contiguous()
        x = x.view(b*subseq_length, 512)




        x_t = self.fcseq_t(x)
        x_q = self.fcseq_q(x)

        x_t = torch.cat([
            fc(x_t) for fc in self.lastfc_t
        ],dim=-1)

        x_q = torch.cat([
            fc(x_q) for fc in self.lastfc_q
        ],dim=-1)



        x = torch.cat([x_t, x_q], dim=-1)


        return xyzwpq_l4, xyzwpq_l3, xyzwpq_l2, xyzwpq_l1, x, None




