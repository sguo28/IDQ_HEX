import torch
import torch.nn as nn


class F_Network(nn.Module):
    def __init__(self):
        super(F_Network,self).__init__()
        ## global state
        self.global_fc = nn.Sequential(nn.Linear(1*54*46,256), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(256,64) , nn.ReLU())
        self.fc2 = nn.Linear(64, 1)

    def forward(self, global_state):
        flattened = torch.flatten(global_state, start_dim=1) # 1*54*46
        global_fc_out = self.global_fc(flattened)
        f_output1 = self.fc1(global_fc_out)
        f_output2 = self.fc2(f_output1)
        return f_output2
