import torch
import torch.nn as nn
import hexagdly
import torch.nn.functional as F

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


#this is to do convolution.
class F_Network_all(nn.Module):
    def __init__(self,input_dim):
        super(F_Network_all,self).__init__()
        ## global state
        self.hexconv_1 = hexagdly.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=3, bias=True)
        # self.hexpool = hexagdly.MaxPool2d(kernel_size=1, stride=2)
        self.hexconv_2 = hexagdly.Conv2d(16, 64, 3, 3)  # 1, 16, 15, 18
        self.global_fc = nn.Sequential((nn.Linear(64*6*6, 256)))  # ,nn.Dropout(0.5)
        ## local state
        self.output_f = nn.Linear(256,1)
        ## concat_fc
        self.cat_fc = nn.Linear(64*6*6+3,256)

    def forward(self, global_state,local_state):
        ## global state
        conv1_out = F.relu(self.hexconv_1(global_state))  # 1, 16, 15, 18
        conv2_out = F.relu(self.hexconv_2(conv1_out))  # 1, 64, 5, 6
        flattened = torch.flatten(conv2_out, start_dim=1) # 1, 64*5*6

        concat_fc = torch.cat((flattened,local_state[:,1:]),dim=1)
        fc_out = F.relu(self.cat_fc(concat_fc))
        f_value = self.output_f(fc_out)

        return f_value
