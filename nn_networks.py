import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)
    
class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,256)
		#self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(256,128)
		#self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim,128)
		#self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		#self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x



class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,256)
		#self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		#self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		#self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = torch.tanh(self.fc4(x))

		action = action * self.action_lim

		return action



class DQN(nn.Module):

	def __init__(self, state_dim, n_action):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(DQN, self).__init__()

		self.state_dim = state_dim
		self.action_dim = n_action

		self.fcs1 = nn.Linear(state_dim,256)
		#self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(256,128)
		#self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fc3 = nn.Linear(128,self.action_dim)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		x = self.fc3(s2)

		return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicAutoEncoderNetwork(nn.Module):
	def __init__(self, obs_dim, action_dim, encoding_dim):
		super(DynamicAutoEncoderNetwork, self).__init__()

		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.encoding_dim = encoding_dim

		### State Encoder ###
		self.encoder = nn.Sequential(
			nn.Linear(obs_dim,256), nn.ReLU(),
			nn.Linear(256,126), nn.ReLU(),
			nn.Linear(126,encoding_dim)
		)

		### State Predictor Given Prvious State and Current Encoded Image and Action ###
		self.gru_hidden_dim = encoding_dim
		self.rnn_layer = nn.GRU(input_size=self.encoding_dim + self.action_dim, hidden_size=self.gru_hidden_dim, batch_first=True) 

		### Image Reconstructed from the State Predictors ###
		self.decoder = nn.Sequential(
			nn.Linear(encoding_dim,126), nn.ReLU(),
			nn.Linear(126,256), nn.ReLU(),
			nn.Linear(256,obs_dim)
		)


if __name__ == '__main__':

	dynautoencoder_nn = DynamicAutoEncoderNetwork(obs_dim=9, action_dim=3, encoding_dim=16)

	batch_obs_stream = torch.FloatTensor(np.random.rand(3, 37, 3, 3)).view(3, 37, -1)
	batch_act_stream = torch.FloatTensor(np.random.rand(3, 37, 3))
	batch_state_est_stream = torch.FloatTensor(np.random.rand(3, 37, 16))

	print('batch_obs_stream', batch_obs_stream.size())

	encoding_streams = dynautoencoder_nn.encoder(batch_obs_stream)

	print('encoding_streams', encoding_streams.size())
	print('batch_act_stream', batch_act_stream.size())
	x_stream = torch.cat([encoding_streams, batch_act_stream], 2)

	print('x_stream', x_stream.size())

	h0 = batch_state_est_stream[:,0,:].unsqueeze(1)

	print('h0', h0.size())

	output, h_n = dynautoencoder_nn.rnn_layer(x_stream, h0.permute(1,0,2))


	print('output', output.size())

	pred_obs = dynautoencoder_nn.decoder(output)

	print('pred_obs', pred_obs.size())