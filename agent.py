from nn_networks import DynamicAutoEncoderNetwork, DQN

import torch
import torch.nn.functional as F

import os, random
import numpy as np



#----------YHJ--------------#
from nn_networks import DQN
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon_greedy = 0.2
discount_rate = 0.9

class DQN_Agent:

    def __init__(self, state_dim):
        self.dqn_network = DQN(state_dim, n_action=6).to(DEVICE)
        self.target_dqn_network = DQN(state_dim, n_action=6).to(DEVICE)
        self.nn_optimizer = torch.optim.Adam(self.dqn_network.parameters())
        self.loss_criterion = torch.nn.MSELoss()

        self._iteration = 0

    def update(self, batch_state, batch_actions, batch_reward, batch_next_state, batch_done):
        # Copied from https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        torch_state_batch = torch.FloatTensor(np.array(batch_state)).to(DEVICE)
        torch_action_batch = torch.LongTensor(np.array(batch_actions)).to(DEVICE).unsqueeze(-1)
        torch_action_batch = torch_action_batch - 1  # [1, 6] -> [0, 5]
        current_Q_values = self.dqn_network(torch_state_batch).gather(1, torch_action_batch)  # Q(s,a)

        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        torch_next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(DEVICE)
        next_max_q = self.target_dqn_network(torch_next_state_batch).detach().max(1)[0]
        #next_Q_values = not_done_mask * next_max_q <--- We are not doing this yet.
        next_Q_values = next_max_q

        # Compute the target of the current Q values
        torch_reward_batch = torch.FloatTensor(np.array(batch_reward)).to(DEVICE)
        target_Q_values = torch_reward_batch + (discount_rate * next_Q_values)

        # Compute Bellman error
        loss = self.loss_criterion(current_Q_values.squeeze(), target_Q_values.squeeze())

        self.nn_optimizer.zero_grad()
        loss.backward()
        self.nn_optimizer.step()

        loss_val = loss.item()

        self._iteration += 1

        #if self._iteration%10 == 0:
        self.target_dqn_network.load_state_dict(self.dqn_network.state_dict())
            #print('target nn update') 
        return loss_val

    def get_exploit_action(self, state):
        torch_state = torch.FloatTensor(np.array(state)).to(DEVICE)
        q_for_all_action = self.target_dqn_network(torch_state).detach()
        action = torch.argmax(q_for_all_action)
        return action.item()+1 # because he uses [1, 6] instead of [0,5]

    def get_explorative_action(self, state, epsilon_action_det, epsilon_greedy_dynamic):
        if epsilon_action_det > epsilon_greedy:
            action = self.get_exploit_action(state)
        else:
            action = random.randint(0,5) + 1
        #print("random: ", action)
        return action # because he uses [1, 6] instead of [0,5]




class DynamicAutoEncoderAgent:
    '''
    Dynamic Auto Encoder
    '''
    def __init__(self):

        obs_dim = 9
        action_dim = 3
        encoding_dim = 9
        
        self.nn_model = DynamicAutoEncoderNetwork(obs_dim, action_dim, encoding_dim).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(self.nn_model.parameters())
        
        self.state = torch.rand(1, 1, encoding_dim).to(DEVICE) #<--- rnn layer h_0 of shape (num_layers * num_directions, batch, hidden_size)

    def predict_batch_images(self, stream_arr, state_est_arr, act_arr):

        n_batch, n_window, n_obs = stream_arr.shape
        n_batch, n_window, n_state = state_est_arr.shape
        n_batch, n_window, n_act   = act_arr.shape

        stream_arr = torch.FloatTensor(stream_arr).to(DEVICE)
        state_est_arr = torch.FloatTensor(state_est_arr).to(DEVICE)
        act_arr = torch.FloatTensor(act_arr).to(DEVICE)

        ### Encoding ###
        encoding_streams = self.nn_model.encoder(stream_arr)
        ### State Predictor ###
        #print('encoding_streams', encoding_streams.size())
        #print('act_arr', act_arr.size())
        x_stream = torch.cat([encoding_streams, act_arr], 2)
        #print('x_stream', x_stream.size())
        
        h0 = state_est_arr[:,0,:].unsqueeze(1).contiguous()
        #print('h0', h0.size())

        output, h_n = self.nn_model.rnn_layer(x_stream, h0.permute(1,0,2))
        # ### Decoding ###
        #print('output', output.size())
        # output = output.squeeze().unsqueeze(-1).unsqueeze(-1)
        pred_obs_stream = self.nn_model.decoder(output)
        #print('pred_obs_stream', pred_obs_stream.size())
        
        return stream_arr, pred_obs_stream
        

    def get_rnn_states(self, image_stream, action_stream):
        ### Encoding ###
        encoding_streams = self.nn_model.encoder(image_stream)

        ### State Predictor ###
        x_stream = torch.cat([encoding_streams, action_stream], 1).unsqueeze(0)
        h0 = torch.zeros(1, 1, self.nn_model.gru_hidden_dim).to(DEVICE)
        rnn_states, h_n = self.nn_model.rnn_layer(x_stream, h0)

        return rnn_states.detach()

    def step(self, observation, action):
        self.nn_model.eval()

        #print('inside step', 'observation', observation.shape)
        #print('inside step', 'action', action.shape)

        ### Encoding ###

        observation = torch.FloatTensor(observation.flatten()).unsqueeze(0).to(DEVICE).detach()
        action = torch.FloatTensor(action).unsqueeze(0).to(DEVICE).detach()
        

        #print('inside step', 'observation', observation.size())
        #print('inside step', 'action', action.size())

        encoding = self.nn_model.encoder(observation).detach()

        #print('inside step', 'encoding', encoding.size())


        ### State Predictor ###
        x = torch.cat([encoding, action], 1).unsqueeze(0)

        #print('inside step', 'x', x.size())

        new_state, hidden = self.nn_model.rnn_layer(x, self.state)

        #print('inside step', 'new_state', new_state.size())

        self.state = new_state.detach()

        del observation, action, encoding, hidden, new_state
        torch.cuda.empty_cache()

        return self.state.detach().cpu().numpy()
        

    def update(self, stream_arr, state_est_arr, act_arr):
        '''
        The system is learned from N_BATCH trajectories sampled from TRAJ_MEMORY and each of them are cropped with the same time WINDOW
        '''
        self.nn_model.train()

        n_batch, n_window, _, _ = stream_arr.shape
        stream_arr = stream_arr.reshape(n_batch, n_window, -1)
        state_est_arr = state_est_arr.squeeze()
        act_arr = act_arr.squeeze()

        ### Predict One Step Future ###        
        tgt_obs_stream, pred_obs_stream = self.predict_batch_images(stream_arr, state_est_arr, act_arr)
        
        
        #### Translate one step the target for calculating loss in prediction
        tgt_obs_stream = tgt_obs_stream[:, 1:, :]
        pred_obs_stream = pred_obs_stream[:, :-1, :]
        
        #### Cross Entropy Loss ###
        loss = torch.mean((pred_obs_stream - tgt_obs_stream)**2)

        ### Update Model ###
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()

        del pred_obs_stream, tgt_obs_stream, loss
        torch.cuda.empty_cache()

        return loss_val


    def save_the_model(self):
        if not os.path.exists('save/'+self.env_name+'/save/dynautoenc/'):
            os.makedirs('save/'+self.env_name+'/save/dynautoenc/')
        f_name = self.name + '_dynautoenc_network_param_' + '_model.pth'
        torch.save(self.nn_model.state_dict(), 'save/'+self.env_name+'/save/dynautoenc/'+f_name)
        #print('DynamicAutoEncoderAgent Model Saved')

    def load_the_model(self):
        f_name = self.name + '_dynautoenc_network_param_' +  '_model.pth'
        self.nn_model.load_state_dict(torch.load('save/'+self.env_name+'/save/dynautoenc/'+f_name))
        #print('DynamicAutoEncoderAgent Model Loaded')