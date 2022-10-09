#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import torch
import torch.nn as nn
import random
from collections import namedtuple
import numpy as np

from environment import *
Transition = namedtuple('Transition', ['state_l', 'continue_t','last_act','best_action' , 'reward', 'next_state_l','done'])


class DQNAgent(object):

    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions= 8,

                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.
        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory

            num_actions (int): The number of the actions

            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
        '''
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape,
                                     mlp_layers=mlp_layers, device=self.device)
        self.target_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape,
                                          mlp_layers=mlp_layers, device=self.device)

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps
        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state_l,continue_t,last_act, best_action, reward, next_state_l,done) = tuple(ts)
        self.feed_memory(state_l, continue_t,last_act,best_action, reward, next_state_l, done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()

    def step(self, state,env):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph
        Args:
            state (numpy.array): current state
        Returns:
            action (int): an action id
        '''
        state_l = env.featurize(state)
        q_values = self.predict(state_l)

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        """
        add the constraint on the action space that a policy level should last for at least 7 days before it decrease
        and it can increase anytime.
        """

        if state['continue_t']<6:
            last_act=int(state['last_act']/10)
            masked_q_values=q_values[last_act:self.num_actions]
            legal_actions=list(range(last_act,self.num_actions))
        else:
            masked_q_values=q_values
            legal_actions = list(range(self.num_actions))
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        action_idx = np.argmax(masked_q_values)
        probs[action_idx] += (1.0 - epsilon)
        best_action = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[best_action], state_l

    def eval_step(self, state,env):
        ''' Predict the action for evaluation purpose.
        Args:
            state (numpy.array): current state
        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        state_l = env.featurize(state)
        q_values = self.predict(state_l)
        if state['continue_t']<6:
            last_act=int(state['last_act'])
            masked_q_values=q_values[last_act:self.num_actions]
            legal_actions=list(range(last_act,self.num_actions))
        else:
            masked_q_values=q_values
            legal_actions =list(range(self.num_actions))
        best_ind = np.argmax(masked_q_values)
        return legal_actions[best_ind]

    def predict(self, mu_sigma):
        q_values = self.q_estimator.predict_nograd(np.expand_dims(mu_sigma, 0))[0]

        return q_values

    def train(self):
        ''' Train the network
        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch,continue_t_batch,last_act_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN)

        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions=[]
        for b in range(self.batch_size):
            if continue_t_batch[b]<6 :
                masked_q_values = q_values_next[b][last_act_batch[b]:self.num_actions]
                legal_actions = list(range(last_act_batch[b], self.num_actions))
            else:
                masked_q_values = q_values_next[b]
                legal_actions=list(range(self.num_actions))
            best_ind = np.argmax(masked_q_values)
            best_actions.append(legal_actions[best_ind])

        # best_actions = np.argmax(q_values_next, axis=1)

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                       self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

    def feed_memory(self, state_l, continue_t,last_act,best_action, reward, next_state_l, done):


        self.memory.save (state_l,continue_t,last_act, best_action, reward, next_state_l, done)

    def set_device(self, device):
        self.device = device
        self.q_estimator.device = device
        self.target_estimator.device = device


class Estimator(object):


    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):

        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.
        Args:
          s (np.ndarray): (batch, state_len)
        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)
        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target
        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss


class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network
        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values
        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state_l, continue_t,last_act,best_action, reward, next_state_l, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state_l, continue_t,last_act,best_action, reward, next_state_l, done)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))


