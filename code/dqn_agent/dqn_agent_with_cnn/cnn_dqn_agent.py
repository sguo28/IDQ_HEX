import os
import random
from collections import defaultdict
from collections import OrderedDict, deque
import numpy as np
import torch
import torch_optimizer as optim
import itertools
import torch.nn as nn
import torch.nn.functional as F
from config.hex_setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, DQN_OUTPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, CNN_SAVE_PATH, SAVING_CYCLE, CNN_RESUME, STORE_TRANSITION_CYCLE, H_AGENT_SAVE_PATH, TERMINAL_STATE_SAVE_PATH,CUDA,NUM_REACHABLE_HEX
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from dqn_agent.replay_buffer import ReplayMemory,Prime_ReplayMemory,F_ReplayMemory
from .cnn_dqn_network import DQN_network, DQN_target_network
from torch.optim.lr_scheduler import StepLR
from dqn_option_agent.option_network import OptionNetwork,TargetOptionNetwork
from dqn_option_agent.f_approx_network import F_Network_all
from torch.utils.tensorboard import SummaryWriter
import time

class DeepQNetworkAgent:
    def __init__(self,hex_diffusion, option_num=0, isoption=False,islocal=True,ischarging=True):
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.charging_dim = CHARGING_DIM
        self.output_dim = DQN_OUTPUT_DIM
        self.log_softmax=torch.nn.LogSoftmax(dim=1)
        self.device =torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
        self.path = CNN_SAVE_PATH

        self.state_feature_constructor = FeatureConstructor()
        self.q_network = DQN_network(self.input_dim, self.output_dim)
        self.target_q_network = DQN_target_network(self.input_dim, self.output_dim)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate,eps=1e-4)

        self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=1000, gamma=0.99) # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        self.h_train_step=0
        self.f_train_step=0
        self.fo_train_step=0
        self.load_network()
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.f_memory=F_ReplayMemory(int(3e5))
        self.fo_memory = F_ReplayMemory(int(1e5))
        self.h_memory=Prime_ReplayMemory(int(6e5))
        self.decayed_epsilon = self.start_epsilon
        self.record_list = []
        self.global_state_dict = OrderedDict()
        self.time_interval = int(0)
        self.global_state_capacity = 14*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.option_dim=option_num
        self.writer = SummaryWriter('logs/results_log_tfboard_{}_{}'.format(self.learning_rate, option_num))
        if option_num>0:
            #use option network
            self.option_dim=option_num
            self.h_network_list = []
            self.h_target_network_list=[]
            self.load_option_networks(self.option_dim)
            self.middle_terminal = self.init_terminal_states()

            self.f_network = F_Network_all(INPUT_DIM)
            # self.load_f_params()
            self.f_network.to(self.device)

            self.fo_network=F_Network_all(INPUT_DIM)
            self.fo_network.to(self.device)

            self.f_lower=np.array([-.1 for _ in range(24)])
            self.f_upper = np.array([.1 for _ in range(24)])
            self.f_median=np.array([0.0 for _ in range(24)])
            self.f_max=np.array([1 for _ in range(24)])
            self.f_median_episode = np.array([0.0 for _ in range(24)])
            self.f_max_episode = np.array([1 for _ in range(24)])

            self.fo_lower=np.array([-.1 for _ in range(24)])
            self.fo_upper = np.array([.1 for _ in range(24)])
            self.fo_median=np.array([0.0 for _ in range(24)])


            self.f_optimizer = torch.optim.Adam(self.f_network.parameters(), lr=1e-4)
            self.fo_optimizer=torch.optim.Adam(self.fo_network.parameters(), lr=1e-4)
            self.h_optimizer = torch.optim.Adam(self.h_network_list[0].parameters(), lr=1e-4)



    def load_option_networks(self,option_num):
        self.h_network_list=[]
        for option_net_id in range(option_num):
            h_network = OptionNetwork(self.input_dim,1+6+5)
            h_target_network=TargetOptionNetwork(self.input_dim, 1+6+5)
            # checkpoint = torch.load('saved_h/ht_network_option_1000_%d.pkl'%(option_net_id))  # lets try the saved networks after the 14th day.
            # h_network.load_state_dict(checkpoint['net'])  # , False
            # h_target_network.load_state_dict(checkpoint['net'])
            self.h_network_list.append(h_network.to(self.device))
            self.h_target_network_list.append(h_target_network.to(self.device))
            print('Successfully load H network {}, total option network num is {}'.format(option_net_id,len(self.h_network_list)))

    def init_terminal_states(self):
        """
        we initial a dict to check the sets of terminal hex ids by hour by option id
        :param oid: ID for option network
        :return:
        """
        middle_terminal = dict()
        for oid in range(self.option_dim):
            with open( 'saved_f/term_states_%d.csv' % oid, 'r') as ts:
                next(ts)
                for lines in ts:
                    line = lines.strip().split(',')
                    hr, hid = line  # option_network_id, hour, hex_ids in terminal state
                    if (oid, int(hr)) in middle_terminal.keys():
                        middle_terminal[(oid, int(hr))].append(int(hid))
                    else:
                        middle_terminal[(oid, int(hr))] = [int(hid)]
        return middle_terminal

    def load_f_params(self):
        checkpoint = torch.load('saved_f/f_network_option_1000_%d.pkl' % (0))  # lets try the saved networks after the 14th day.
        self.f_network.load_state_dict(checkpoint['net'])  # , False
        print('Successfully load f network {}, total option network num is {}'.format(0,1))


    def reset_f_storage(self):
        self.f_median_all=[[] for _ in range(24)]
        self.fo_median_all = [[] for _ in range(24)]
        self.f_all=np.zeros((24*60,NUM_REACHABLE_HEX))
        self.fo_all = np.zeros((24 * 60, NUM_REACHABLE_HEX))
        self.f_sliding=deque(maxlen=NUM_REACHABLE_HEX*15) #track 15 ticks


    def summarize_median(self):
        self.f_median_episode=np.array([np.median(val) for val in self.f_median_all])
        minf=np.array([np.min(val) for val in self.f_median_all])
        maxf=np.array([np.max(val) for val in self.f_median_all])
        self.f_max_episode=np.array([np.maximum(abs(minv-medv),abs(maxv-medv)) for minv,maxv,medv in zip(minf,maxf,self.f_median_episode)])


    def get_local_f(self,global_state,tick):
        with torch.no_grad():
            local_state=[]
            for i in range(NUM_REACHABLE_HEX):
                state=[tick,i,1]
                local_state.append(self.state_feature_constructor.construct_state_features(state))

            #use state[0] for hex id if after feature construction, otherwise use state[1]
            hex_diffusions = np.array([np.tile(self.hex_diffusion[state[0]], (1, 1, 1)) for state in
                                       local_state])  # state[1] is hex_id
            local_state=torch.from_numpy(np.array(local_state)).to(dtype=torch.float32, device=self.device)
            g_state=torch.from_numpy(np.concatenate([np.tile(global_state,(len(local_state),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device)
            f_vals=self.get_f_value(local_state,g_state).cpu().numpy()
            self.f_sliding+=f_vals.tolist() #sliding window to track the f values
            fo_vals=self.fo_network.forward(g_state,local_state).cpu().numpy()
            hrs=int(tick//3600%24)
            print('length of local_state',len(f_vals),'Setting local values before:', self.f_median[hrs], self.f_lower[hrs], self.f_upper[hrs])
            self.f_median[hrs]=np.median(self.f_sliding)
            minf=np.min(self.f_sliding); maxf=np.max(self.f_sliding)
            self.f_max[hrs]=np.maximum(abs(minf-self.f_median[hrs]),abs(maxf-self.f_median[hrs])) #set the maximum of the hours
            self.current_max_f=self.f_max[hrs]

            print('The median of fvals is:',np.median(f_vals),np.median(fo_vals))
            self.f_median_all[hrs]+=f_vals.tolist()
            self.fo_median_all[hrs] += fo_vals.tolist()
            self.current_f=f_vals-self.f_median[hrs]
            self.f_lower[hrs]=np.percentile(f_vals,20)
            self.f_upper[hrs]=np.percentile(f_vals,80)
            print('Setting local values after:', self.f_median[hrs],self.f_lower[hrs],self.f_upper[hrs])

    def is_terminal(self,local_state,global_state,original_states):
        with torch.no_grad():
            original_states=np.array(original_states)
            f_vals=self.get_f_value(local_state,global_state).cpu().numpy()
            hrs=original_states[:,0]//3600%24
            hrs=hrs.astype(int)
            terminal=np.array(   ((f_vals<self.f_lower[hrs]) | (f_vals>self.f_upper[hrs]))   )
            return terminal[:,0],f_vals

    def is_local_terminal(self,local_state,global_state):
        state_reps = np.array([self.state_feature_constructor.construct_state_features(state) for state in local_state])
        hex_diffusions = np.array([np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in
                                   local_state])  # state[1] is hex_id
        terminal,fvals=self.is_terminal(torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.concatenate([np.tile(global_state,(len(state_reps),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device),local_state) # if the state is considered as terminal, we dont use it..

        print('N f values ={}, mean ={}, 75 percentile={}, 25 percentile={}'.format(fvals.shape[0],np.mean(fvals),np.percentile(fvals,75),np.percentile(fvals,25)))

        return terminal


    # def is_terminal(self,states,oid):
    #     """
    #     :param states:
    #     :return: a list of bool
    #     """
    #
    #     return [True if state[1] in self.middle_terminal[(oid,int(state[0] // (60 * 60) % 24))] else False for state in states]

    def load_network(self):
        if CNN_RESUME:
            lists = os.listdir(self.path)
            lists.sort(key=lambda fn: os.path.getmtime(self.path + "/" + fn))
            newest_file = os.path.join(self.path, lists[-1])
            path_checkpoint = newest_file  #'logs/test/cnn_dqn_model/duel_dqn_69120.pkl'  #
            checkpoint = torch.load(path_checkpoint)

            self.q_network.load_state_dict(checkpoint['net'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            self.train_step = checkpoint['step']
            self.copy_parameter()
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Successfully load saved network starting from {}!'.format(str(self.train_step)))



    def get_action_mask(self, batch_state, batch_valid_action):
        """
        the action space: the first 3 is for h_network slots, then 7 relocation actions,and 5 nearest charging stations.
        :param batch_state: state
        :param batch_valid_action: info that limites to relocate to reachable neighboring hexes
        :return:
        """
        mask = np.zeros((len(batch_state), self.output_dim),dtype=bool)  # (num_state, 15)
        for i, state in enumerate(batch_state):
            mask[i][(self.option_dim+ batch_valid_action[i]):(self.option_dim+self.relocation_dim)] = 1  # limited to relocate to reachable neighboring hexes
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][(self.option_dim+self.relocation_dim):] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:(self.option_dim+self.relocation_dim)] = 1  # no relocation, must charge
        return mask

    def get_option_mask(self,local_state,global_state,original_state):
        """
        self.is_terminal is to judge if the state is terminal state with the info of hour and hex_id
        :param states:
        :return:
        """
        terminate_option_mask = np.zeros((len(local_state),self.output_dim),dtype=bool)
        for option in range(self.option_dim):
            terminal,_=self.is_terminal(local_state,global_state,original_state)
            terminate_option_mask[:,option] = terminal # set as 0 if not in terminal set
        # for oid in range(self.option_dim,self.option_dim):
        #     terminate_option_mask[:,oid] = 1 # mask out empty options
        return terminate_option_mask

    def get_actions(self, states, num_valid_relos, global_state,assigned_option_ids):
        """
        :param global_states:
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return:
        """
        with torch.no_grad():
            self.decayed_epsilon = max(self.final_epsilon, (self.start_epsilon - self.train_step * (
                    self.start_epsilon - self.final_epsilon) / self.epsilon_steps))
            state_reps = np.array([self.state_feature_constructor.construct_state_features(state) for state in states])
            hex_diffusions = np.array([np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in
                              states])  # state[1] is hex_id
            mask = self.get_action_mask(states, num_valid_relos)  # mask for unreachable primitive actions
            option_mask = self.get_option_mask(torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.concatenate([np.tile(global_state,(len(states),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device),states) # if the state is considered as terminal, we dont use it..

            if random.random() > self.decayed_epsilon:  # epsilon = 0.1
                full_action_values = self.q_network.forward(
                    torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.concatenate([np.tile(global_state,(len(states),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device))
                assigned_option_ids=np.array(assigned_option_ids,dtype=int)
                #choose an action, in the following conditions:
                #1. if it is a terminal state, select the one with the maximum Q value
                #2  if it is not a terminal state, following the previous option policy to get the action (must)
                # ---- if there is no previous options, randomly select a policy with the largest value
                update_idx=assigned_option_ids>-1 #those states who are under options
                full_action_values[
                    np.arange(full_action_values.shape[0])[update_idx], assigned_option_ids[update_idx]] = 9e10
                # print('take a look at processed mask {}'.format(mask))
                terminate_option_mask = torch.from_numpy(option_mask).to(dtype=torch.bool, device=self.device)
                full_action_values[
                    terminate_option_mask] = -9e10  # if the chosen policy is in a terminal state, mask it out
                full_action_values[torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)] = -9e10

                #lets do 50% option and 50% premitive action
                if self.option_dim>0:
                    full_action_values[:,0]=-9e10

                # action_indexes = torch.argmax(full_action_values, dim=1).cpu().numpy()
                log_softmax = torch.softmax(full_action_values, dim=1)
                action_indexes = torch.flatten(torch.multinomial(log_softmax, 1)).cpu().numpy() # choose one action

            else:
                full_action_values = np.random.random(
                    (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                assigned_option_ids=np.array(assigned_option_ids,dtype=int)
                update_idx=assigned_option_ids>-1


                #exploration based on f values, choose the large ones.
                if self.option_dim>0:
                    full_action_values = -9e10*np.ones(
                        (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                    hex_ids=np.array(states)[:,1].astype(int)
                    current_idx = self.neighbor_id[hex_ids]
                    self_f=np.abs(self.current_f[current_idx].reshape(len(current_idx), 7))

                    full_action_values[:,1:8] = -np.abs(self_f-self.current_max_f)   #np.minimum(self_f,np.abs(self_f-self.current_max_f))  # get all the f values

                #lets do 50% option and 50% premitive action
                # if self.option_dim>0:
                #     full_action_values[:,0]=9e10

                # full_action_values[np.arange(full_action_values.shape[0])[update_idx], assigned_option_ids[
                #     update_idx]] = 9e10  # must choose

                full_action_values[option_mask] = -9e10 #if the chosen policy is in a terminal state, mask it out
                full_action_values[mask] = -9e10 #choose other actions
                if self.option_dim>0:
                    full_action_values[:,0]=-9e10


                log_softmax = torch.softmax(torch.from_numpy(full_action_values).to(dtype=torch.float32, device=self.device), dim=1)
                action_indexes = torch.flatten(torch.multinomial(log_softmax, 1)).cpu().numpy() # choose one action

                # action_indexes = np.argmax(full_action_values, 1)

        selected_actions=list(action_indexes)  #this is the set of actions selected by DQN, which will be used for training
        converted_action_indexes, new_assigned_opts = self.convert_option_to_primitive_action_id_f(np.array(action_indexes), state_reps,
                                                                                             global_state,
                                                                                             hex_diffusions, mask)
        action_to_execute=converted_action_indexes-self.option_dim
        #start with false
        contd_options = np.zeros(len(action_indexes), dtype=bool)
        for idx,opts in enumerate(zip(new_assigned_opts,assigned_option_ids)):
            new_opt,old_opt=opts
            if new_opt==old_opt and new_opt>=0:
                contd_options[idx]=True
        #returning the identified actions, assigned options, and if it is a continuing option of a new option
        return selected_actions,action_to_execute,new_assigned_opts, contd_options

    def convert_option_to_primitive_action_id(self, action_indexes, state_reps, global_state, hex_diffusions, mask):
        """
        we convert the option ids, e.g., 0,1,2 for each h network, to the generated primitive action ids
        :param action_indexes:
        :param state_reps:
        :param global_state:
        :param hex_diffusions:
        :param mask:
        :return:
        """
        with torch.no_grad():
            assigned_options = -np.ones(len(action_indexes))
            ids_require_option = defaultdict(list)
            for id, action_id in enumerate(action_indexes):
                if action_id < self.option_dim:
                    ids_require_option[action_id].append(id)

            converted_states=torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32,device=self.device)
            global_states=torch.from_numpy(np.concatenate([np.tile(global_state,(len(state_reps),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device)
            all_mask=torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)

            option_generated=[]
            for option_id in range(self.option_dim):
                if ids_require_option[option_id]:
                    full_option_values = self.h_target_network_list[option_id].forward(converted_states,global_states)
                    full_option_values=full_option_values[ids_require_option[option_id]]
                    # here mask is of batch x 15 dimension, we omit the first 3 columns, which should be options.
                    primitive_action_mask = all_mask[ids_require_option[option_id],
                                            self.option_dim:]  # only primitive actions in option generator
                    full_option_values[primitive_action_mask] = -9e10
                    # full_option_values[:,0]=-9e10 #no self relocation

                    #all negative values for H at the location
                    acts=torch.argmax(full_option_values, dim=1)
                    option_generated.append(acts)
                    #lets try a softmax implementation
                    # log_softmax=torch.softmax(full_option_values,dim=1)
                    # actions=torch.flatten(torch.multinomial(log_softmax,1)) #choose one action
                    # option_generated.append(actions)
                else:
                    option_generated.append(None)

            for option_id in range(self.option_dim):
                if ids_require_option[option_id]:
                    option_generated_premitive_action_ids = option_generated[option_id].cpu().numpy()  # let option network select primitive action
                    action_indexes[ids_require_option[option_id]] = option_generated_premitive_action_ids + self.option_dim  # 12 to 15
                    assigned_options[ids_require_option[option_id]] = option_id
                    # cover the option id with the generated primitive action id
            # for a,o in zip(action_indexes,assigned_options):
            #     if o>-1 and a==self.option_dim:
            #         print('wrong results in cnn_dqn_Agent line 258',o,a)
            #     if a<self.option_dim:
            #         print('wrong results in cnndqnagent line 258',o,a)
        return action_indexes, assigned_options


    def convert_option_to_primitive_action_id_f(self, action_indexes, state_reps, global_state, hex_diffusions, mask):
        """
        we convert the option ids, e.g., 0,1,2 for each h network, to the generated primitive action ids
        :param action_indexes:
        :param state_reps:
        :param global_state:
        :param hex_diffusions:
        :param mask:
        :return:
        """
        with torch.no_grad():
            assigned_options = -np.ones(len(action_indexes))
            ids_require_option = defaultdict(list)
            for id, action_id in enumerate(action_indexes):
                if action_id < self.option_dim:
                    ids_require_option[action_id].append(id)

            hex_ids=state_reps[:,0].astype(int) #all hex ids

            option_generated=[]
            for option_id in range(self.option_dim):
                if ids_require_option[option_id]:
                    current_idx=self.neighbor_id[hex_ids]
                    full_option_values = self.current_f[current_idx].reshape(len(current_idx),7)#get all the f values
                    # print('shape of idx',current_idx.shape, 'shape of full option values', full_option_values.shape)
                    current_f=self.current_f[hex_ids]
                    # print('shape of curren tf',current_f.shape)
                    full_option_values=np.abs(full_option_values)
                    full_option_values=full_option_values[ids_require_option[option_id]]
                    # here mask is of batch x 15 dimension, we omit the first 3 columns, which should be options.
                    primitive_action_mask = mask[ids_require_option[option_id],
                                            self.option_dim:self.option_dim+7]  # only primitive actions in option generator
                    full_option_values[primitive_action_mask] = -9e10
                    # full_option_values[:,0]=-9e10 #no self relocation

                    #all negative values for H at the location
                    acts=np.argmax(full_option_values, axis=1)
                    option_generated.append(acts)
                    #lets try a softmax implementation
                    # log_softmax=torch.softmax(full_option_values,dim=1)
                    # actions=torch.flatten(torch.multinomial(log_softmax,1)) #choose one action
                    # option_generated.append(actions)
                else:
                    option_generated.append(None)

            for option_id in range(self.option_dim):
                if ids_require_option[option_id]:
                    option_generated_premitive_action_ids = option_generated[option_id]  # let option network select primitive action
                    action_indexes[ids_require_option[option_id]] = option_generated_premitive_action_ids + self.option_dim  # 12 to 15
                    assigned_options[ids_require_option[option_id]] = option_id
                    # cover the option id with the generated primitive action id
            # for a,o in zip(action_indexes,assigned_options):
            #     if o>-1 and a==self.option_dim:
            #         print('wrong results in cnn_dqn_Agent line 258',o,a)
            #     if a<self.option_dim:
            #         print('wrong results in cnndqnagent line 258',o,a)
        return action_indexes, assigned_options



    def add_global_state_dict(self, global_state_list):
        for tick in global_state_list.keys():
            if tick not in self.global_state_dict.keys():
                self.global_state_dict[tick] = global_state_list[tick]
        if len(self.global_state_dict.keys()) > self.global_state_capacity: #capacity limit for global states
            for _ in range(len(self.global_state_dict.keys())-self.global_state_capacity):
                self.global_state_dict.popitem(last=False)


    def add_transition(self, state, action, next_state, reward, terminate_flag, time_steps, valid_action):
        self.memory.push(state, action, next_state, reward, terminate_flag, time_steps, valid_action)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples
        # state, action, next_state, reward = zip(*samples)
        # return state, action, next_state, reward

    def add_H_transition(self, state, action, next_state, trip_flag, time_steps, valid_action):
        self.h_memory.push(state, action, next_state, trip_flag, time_steps, valid_action)

    def H_batch_sample(self):
        samples = self.h_memory.sample(self.batch_size//2)  # random.sample(self.memory, self.batch_size)
        return samples

    def f_batch_sample(self):
        samples = self.f_memory.sample(self.batch_size//2)  # random.sample(self.memory, self.batch_size)
        return samples

    def add_f_transition(self,data):
        #data is of format [state, state_]
        self.f_memory.push(data[0],data[1])

    def add_fo_transition(self,data):
        #data is of format [state, state_]
        self.fo_memory.push(data[0],data[1])

    def get_f_value(self,local_state,global_state):
        return self.f_network.forward(global_state,local_state)

    def get_main_Q(self, local_state, global_state):
        return self.q_network.forward(local_state, global_state)

    def get_target_Q(self, local_state, global_state):
        return self.target_q_network.forward(local_state, global_state)

    def copy_parameter(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def copy_H_parameter(self):

        self.h_target_network_list[0].load_state_dict(self.h_network_list[0].state_dict())

    def soft_target_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    # def soft_f_update(self, tau):
    #     """Soft update model parameters.
    #     θ_target = τ*θ_local + (1 - τ)*θ_target
    #     Params
    #     ======
    #         local_model (PyTorch model): weights will be copied from
    #         target_model (PyTorch model): weights will be copied to
    #         tau (float): interpolation parameters
    #     """
    #     self.f_network.load_state_dict(self.target_f_network.state_dict())

    def train(self, record_hist):
        self.train_step += 1
        # print('Main buffer = {}, h buffer={}, f buffer={}'.format(len(self.memory.memory),len(self.h_memory.memory),len(self.f_memory.memory)))
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return

        transitions = self.batch_sample()
        batch = self.memory.Transition(*zip(*transitions))

        global_state_reps = [self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state]  # should be list of np.array

        global_next_state_reps = [self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]  # should be list of np.array

        state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
                           batch.next_state]

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]],(1,1,1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]],(1,1,1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        terminal_flag= torch.from_numpy(np.array(batch.terminate_flag)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(np.concatenate([np.array(global_state_reps),np.array(hex_diffusion)],axis=1)).to(dtype=torch.float32, device=self.device)
        global_next_state_batch = torch.from_numpy(np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)],axis=1)).to(dtype=torch.float32,
                                                                                        device=self.device)

        # print('Any weired actions?', action_batch[action_batch>=self.output_dim])
        # print('Any weired actions?', action_batch[action_batch<0])
        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())
        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch)
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        option_mask = self.get_option_mask(next_state_batch,global_next_state_batch,batch.next_state)
        all_q_[torch.from_numpy(option_mask).to(dtype=torch.bool,device=self.device)] = -9e10
        all_q_[torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)] = -9e10
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)
        print('Max Q={}, Max target Q={},Gamma={}'.format(torch.max(q_state_action),torch.max(maxq),self.gamma))
        if self.option_dim==0:
            y = reward_batch + (1-terminal_flag)*maxq*torch.pow(self.gamma,time_step_batch)
        else:
            hrs = [state[0] // 3600 % 24 for state in batch.state]
            hrs_ = [state[0] // 3600 % 24 for state in batch.next_state]
            f_median = torch.from_numpy(self.f_median_episode[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
            f_max= torch.from_numpy(self.f_max_episode[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
            # fo_median = torch.from_numpy(self.fo_median[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
            f_s_ = self.f_network.forward(global_next_state_batch, next_state_batch).detach()
            y = reward_batch + (1 - terminal_flag) * maxq * torch.pow(self.gamma, time_step_batch)
            # y = reward_batch - 0.5*abs(f_s_-f_median)+ (1 - terminal_flag) * maxq * torch.pow(self.gamma, time_step_batch)                                                                                   time_step_batch)
        # yz=reward_batch + (1-terminal_flag)*maxq*torch.pow(self.gamma,time_step_batch)

        loss = F.smooth_l1_loss(q_state_action, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
        self.optimizer.step()
        # self.lr_scheduler.step()
        self.writer.add_scalar('main_dqn/train_loss',loss, self.train_step)
        self.writer.add_scalar('main_dqn/maxQ', y.mean(), self.train_step)
        # self.writer.add_scalar('main_dqn/option_percent', y.mean(), self.train_step)
        # self.record_list.append([self.train_step, round(float(loss),3), round(float(reward_batch.view(-1).mean()),3),self.optimizer.state_dict()['param_groups'][0]['lr'],batch.reward[0], batch.state[0][-1]])
        print('Training step is {}; Learning rate is {}; Epsilon is {}; Average loss is {:.3f}, number of topion used={}'.format(self.train_step,self.lr_scheduler.get_lr(),round(self.decayed_epsilon,4),loss,self.option_dim))


    def train_f(self):
        #retrain the f network
        self.f_network = F_Network_all(INPUT_DIM)
        self.f_network.to(self.device)
        self.f_optimizer = torch.optim.Adam(self.f_network.parameters(), lr=1e-4)
        max_f_size=min(len(self.f_memory.memory),250000)
        # random.shuffle(self.f_memory.memory)  # random shuffle
        if len(self.f_memory)>256:
            for epi in range(1):
                batch_size = 256
                for i in range(0,len(self.f_memory.memory)-batch_size,batch_size):
                    self.f_train_step += 1
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    # transitions=self.f_memory.sample(batch_size)
                    transitions = self.f_memory.memory[i:i+batch_size]
                    sample_batch = self.f_memory.Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array
                    state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                          sample_batch.state]
                    next_state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                               sample_batch.next_state]
                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                              dtype=torch.float32)
                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)
                    f_values=self.f_network.forward(global_state_batch,state_batch)
                    f_values_=self.f_network.forward(global_next_state_batch,next_state_batch)
                    # print(f_values.mean(),f_values.max(),f_values.min())
                    eta = .1 # lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                    delta = 1 #according to Yifan Wang et al. 2019
                    f1=1


                    # loss1 = (0.5 * (f_values_ - f_values).pow(2) + eta * (
                    #             (f_values_ ** 2 - delta) * (f_values ** 2 - delta) + 2 * (
                    #                 f1 ** 2) * f_values_ * f_values + (f1 ** 2 - delta) ** 2))
                    # loss = (0.5 *(f_values_ - f_values).pow(2) + eta*((f_values_ - delta)*(f_values-delta) + f_values_.pow(2)*f_values.pow(2))).mean()  # + (f_values-f_values_).mean())

                    # loss = (0.5 * (f_values_ - f_values).pow(2) + eta * ((f_values_**2 - delta)*(f_values**2-delta) + 1*(f1**2)*f_values_*f_values)).mean()  # + (f_values-f_values_).mean())
                    loss = (0.5 * (f_values_ - f_values).pow(2) + eta * (
                        (f_values_*f_values+f1**2)**2-(f_values**2+f1**2)-(f_values_**2+f1**2))+delta**2).mean()
                    print('F training loss for step {} is {:.3f}'.format(i,loss))
                    self.f_optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.f_network.parameters(), 1)
                    self.f_optimizer.step()
                    self.writer.add_scalar('f_train/train_loss', loss, self.f_train_step)
                    #release the memory

    def train_fo(self):
        #retrain the f network
        self.fo_network = F_Network_all(INPUT_DIM)
        self.fo_network.to(self.device)
        self.fo_optimizer = torch.optim.Adam(self.fo_network.parameters(), lr=1e-4)
        if len(self.fo_memory)>256:
            for epi in range(1):
                batch_size = 256
                for i in range(0,400):
                    self.fo_train_step += 1
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    # transitions = self.fo_memory.memory[i:i+batch_size]
                    transitions=self.fo_memory.sample(batch_size)
                    sample_batch = self.fo_memory.Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array
                    state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                          sample_batch.state]
                    next_state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                               sample_batch.next_state]
                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                              dtype=torch.float32)
                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)
                    f_values=self.fo_network.forward(global_state_batch,state_batch)
                    f_values_=self.fo_network.forward(global_next_state_batch,next_state_batch)
                    # print(f_values.mean(),f_values.max(),f_values.min())
                    eta = 1 # lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                    delta = 0.2 #according to Yifan Wang et al. 2019
                    f1=-((delta /NUM_REACHABLE_HEX)**0.5) #vaue in f(1)


                    # loss1 = (0.5 * (f_values_ - f_values).pow(2) + eta * (
                    #             (f_values_ ** 2 - delta) * (f_values ** 2 - delta) + 2 * (
                    #                 f1 ** 2) * f_values_ * f_values + (f1 ** 2 - delta) ** 2))
                    # loss = (0.5 *(f_values_ - f_values).pow(2) + eta*((f_values_ - delta)*(f_values-delta) + f_values_.pow(2)*f_values.pow(2))).mean()  # + (f_values-f_values_).mean())

                    loss = (0.5 * (f_values_ - f_values).pow(2) + eta * ((f_values_**2 - delta)*(f_values**2-delta) + 2*(f1**2)*f_values_*f_values+(f1**2-delta)**2)).mean()  # + (f_values-f_values_).mean())

                    print('Fo training loss for step {} is {:.3f}'.format(i,loss))
                    self.fo_optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.f_network.parameters(), 1)
                    self.fo_optimizer.step()
                    self.writer.add_scalar('fo_train/train_loss', loss, self.fo_train_step)



    def record_f_threshold(self):
        # find the mean threshold and percentile values for each hour
        vals=[[] for _ in range(24)]
        # self.f_median=[]
        # self.f_lower = []
        # self.f_higher = []
        random.shuffle(self.f_memory.memory)  # random shuffle
        with torch.no_grad():
            if len(self.f_memory)>1024:
                batch_size = 256
                for i in range(0,500):
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    transitions = self.f_memory.sample(batch_size)
                    sample_batch = self.f_memory.Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array
                    state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                          sample_batch.state]
                    next_state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                               sample_batch.next_state]
                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                              dtype=torch.float32)
                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)
                    f_values=self.f_network.forward(global_state_batch,state_batch).cpu().numpy()
                    f_values_=self.f_network.forward(global_next_state_batch,next_state_batch).cpu().numpy()
                    for state,f in zip(sample_batch.state,f_values):
                        hr=(state[0]//3600)%24
                        vals[int(hr)].append(f)
                    for state,f in zip(sample_batch.next_state, f_values_):
                        hr = (state[0] // 3600) % 24
                        vals[int(hr)].append(f)

        self.f_median=0*self.f_median+1*np.array([np.median(v) for v in vals])
        self.f_lower=0*self.f_lower+1*np.array([np.percentile(v,35) for v in vals])
        self.f_upper=0*self.f_upper+1*np.array([np.percentile(v, 65) for v in vals])
        # self.f_mid_lower=np.array([np.percentile(v,35) for v in vals])
        # self.f_mid_upper = np.array([np.percentile(v, 65) for v in vals])


    def record_fo_threshold(self):
        # find the mean threshold and percentile values for each hour
        vals=[[] for _ in range(24)]
        # self.f_median=[]
        # self.f_lower = []
        # self.f_higher = []
        random.shuffle(self.fo_memory.memory)  # random shuffle
        with torch.no_grad():
            if len(self.f_memory)>1024:
                batch_size = 256
                for i in range(0,500):
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    transitions = self.fo_memory.sample(batch_size)
                    sample_batch = self.fo_memory.Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array
                    state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                          sample_batch.state]
                    next_state_reps = [self.state_feature_constructor.construct_state_features(state) for state in
                                               sample_batch.next_state]
                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                              dtype=torch.float32)
                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)
                    f_values=self.fo_network.forward(global_state_batch,state_batch).cpu().numpy()
                    f_values_=self.fo_network.forward(global_next_state_batch,next_state_batch).cpu().numpy()
                    for state,f in zip(sample_batch.state,f_values):
                        hr=(state[0]//3600)%24
                        vals[int(hr)].append(f)
                    for state,f in zip(sample_batch.next_state, f_values_):
                        hr = (state[0] // 3600) % 24
                        vals[int(hr)].append(f)

        self.fo_median=0*self.fo_median+1*np.array([np.median(v) for v in vals])
        self.fo_lower=0*self.fo_lower+1*np.array([np.percentile(v,20) for v in vals])
        self.fo_upper =0*self.fo_upper+1*np.array([np.percentile(v, 80) for v in vals])
        # self.f_mid_lower=np.array([np.percentile(v,35) for v in vals])
        # self.f_mid_upper = np.array([np.percentile(v, 65) for v in vals])


    def reset_h(self):

        self.h_network_list = [OptionNetwork(self.input_dim, 1 + 6 + 5).to(device=self.device)]
        self.h_target_network = [TargetOptionNetwork(self.input_dim, 1 + 6 + 5).to(device=self.device)]
        self.h_optimizer = torch.optim.Adam(self.h_network_list[0].parameters(), lr=1e-3,eps=1e-4)


    def train_h(self):
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return
        self.h_train_step+=1
        transitions = self.H_batch_sample()
        batch = self.h_memory.Transition(*zip(*transitions))



        hrs=[state[0]//3600%24 for state in batch.state]
        hrs_=[state[0]//3600%24 for state in batch.next_state]

        global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state])  # should be list of np.array

        global_next_state_reps = np.array([self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]) # should be list of np.array

        next_zones = np.array([state_[1] for state_ in batch.next_state])  # zone id for choosing actions

        state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
                           batch.next_state]

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]], (1, 1, 1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)

        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)

        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        trip_flag=torch.from_numpy(np.array(batch.trip_flag)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(dtype=torch.float32,
                                                                                               device=self.device)
        global_next_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
            dtype=torch.float32, device=self.device)

        f_median=torch.from_numpy(self.f_median[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        fo_median = torch.from_numpy(self.fo_median[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)

        # f_lower=torch.from_numpy(self.f_lower[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # f_lower_ = torch.from_numpy(self.f_lower[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # f_upper = torch.from_numpy(self.f_upper[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # f_upper_ = torch.from_numpy(self.f_upper[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)

        f_s=self.f_network.forward(global_state_batch,state_batch).detach()
        f_s_ = self.f_network.forward(global_next_state_batch, next_state_batch).detach()
        #
        fo_s=self.fo_network.forward(global_state_batch,state_batch).detach()
        fo_s_ = self.fo_network.forward(global_next_state_batch, next_state_batch).detach()


        # middle_terminal_flag = (((f_s<f_lower) & (f_s>f_upper))).to(dtype=torch.float32,device=self.device)
        # middle_next_terminal_flag = (((f_s_<f_lower_) & (f_s_>f_upper_)) ).to(dtype=torch.float32,device=self.device)

        # middle_next_terminal_flag = torch.from_numpy(np.array((self.is_middle_terminal(batch.next_state)))).unsqueeze(1).to(
        #     dtype=torch.float32, device=self.device)

        q_state_action = self.h_network_list[0].forward(state_batch, global_state_batch).gather(1, action_batch.long())

        all_q_ = self.h_target_network_list[0].forward(next_state_batch, global_next_state_batch).detach()#lets change this to global state batch and see if error continues
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        all_q_[mask[:,self.option_dim:]] = -9e10
        maxq = all_q_.max(1)[0].unsqueeze(1)

        #
        # f_s=f_s-f_median #convert to 0 median
        # f_s_=f_s_-f_median #convert to 0 median
        # fo_s=fo_s-fo_median
        # fo_s_=fo_s_-fo_median
        # f_s=torch.from_numpy(f_s).to(dtype=torch.float32,device=self.device);f_s_=torch.from_numpy(f_s_).to(dtype=torch.float32,device=self.device)
        #

        # pseudo_reward = torch.abs(f_s)-torch.abs(f_s_)

        #this following is for relocating from middle to peripheral
        # pseudo_reward = torch.abs(f_s_)-torch.abs(f_s)-0.2   #+2*(torch.abs(fo_s)-torch.abs(fo_s_))-1

        pseudo_reward=torch.abs(f_s_)-1
        # r1=torch.abs(f_s_)/(1+torch.abs(fo_s_))
        # r2=(torch.abs(f_s))/(1+torch.abs(fo_s))
        #
        # pseudo_reward=r1-r2-0.1
        #
        #
        # # pseudo_reward[pseudo_reward < 0] = pseudo_reward[pseudo_reward < 0] * 5  # penaltize moving back
        # pseudo_reward=torch.abs(f_s_)-torch.abs(f_s)
        # pseudo_reward+=0.02* trip_flag #bonus for matching
        # pseudo_reward+=trip_flag*0.1
        # pseudo_reward+=middle_next_terminal_flag*5
        #use demand and supply gap to update hex_zones

        # hex_rows=self.hex_xy[next_zones,0] #find the rows.
        # hex_cols=self.hex_xy[next_zones,1] #find the columns
        # demand_supply_gap=global_state_reps[:,0,:]/5-global_state_reps[:,1,:]
        # gap_reward=demand_supply_gap[np.arange(len(hex_rows)),hex_rows,hex_cols]
        # #
        # pseudo_reward+=0.01*torch.from_numpy(gap_reward).unsqueeze(1).to(dtype=torch.float32, device=self.device) #demand supply gap

        y =(pseudo_reward + (1-trip_flag)*0.95*maxq)
       #  y = pseudo_reward #
        # print(middle_terminal_flag.shape, f_s.shape, pseudo_reward.shape, maxq.shape, y.shape)
        loss = F.smooth_l1_loss(q_state_action,y)
        print('H network training loss for step {} is {}'.format(self.train_step,loss))
        #
        # if self.train_step%100==0:
        #     print('Max Q={}, Max target Q={}, Loss = {}, Gamma={}, mean diff={}, mean reward={}'.format(torch.max(q_state_action), torch.max(maxq),loss, self.gamma,mean_diff,mean_pseudo))
        #     print('Mean of main f={}, mean of target f={}'.format(f_s.mean(),f_s_.mean()))
        #     with open('saved_h/ht_train_log_{}.csv'.format(self.num_option),'a') as f:
        #         f.writelines('Train step={}, Max Q={}, Max target Q={}, Loss = {}, Mean f_diff={}, Mean pseudo-reward={}\n'.format(self.train_step,torch.max(q_state_action), torch.max(maxq),loss,mean_diff,mean_pseudo))
        self.writer.add_scalar('h_train/train_loss', loss, self.h_train_step)
        self.writer.add_scalar('h_train/max_q', y.mean(), self.h_train_step)
        self.h_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.h_network_list[0].parameters(), self.clipping_value)
        self.h_optimizer.step()
        # self.lr_scheduler.step()

    def save_parameter(self):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.option_dim>0:
            checkpoint = {
                "net_dqn": self.q_network.state_dict(),
                "net_f":self.f_network.state_dict(),
                "net_h":self.h_target_network_list[0].state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
                "lr_scheduler": self.lr_scheduler.state_dict()
            }
        else:
            checkpoint = {
                "net_dqn": self.q_network.state_dict(),
                "net_f":[],
                "net_h":[],
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
                "lr_scheduler": self.lr_scheduler.state_dict()
            }

        if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
        torch.save(checkpoint, 'logs/test/cnn_dqn_model/dqn_fh_{}_{}_{}_{}.pkl'.format(self.learning_rate,self.option_dim,bool(self.local_matching),str(self.train_step)))
            # record training process (stacked before)

    # def get_action_mask(self, batch_state, batch_valid_action):
    #     mask = np.zeros([len(batch_state), self.output_dim])
    #     for i, state in enumerate(batch_state):
    #         mask[i][batch_valid_action[i]:self.relocation_dim] = 1
    #         # here the SOC in state is still continuous. the categorized one is in state reps.
    #         if state[-1] > HIGH_SOC_THRESHOLD:
    #             mask[i][self.relocation_dim:] = 1  # no charging, must relocate
    #         elif state[-1] < LOW_SOC_THRESHOLD:
    #             mask[i][:self.relocation_dim] = 1  # no relocation, must charge
    #
    #     mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
    #     return mask
