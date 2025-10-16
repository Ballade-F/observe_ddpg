import torch
import torch.nn as nn
import numpy as np
from replay_buffer import ReplayBuffer
from torch.distributions import Beta, Normal

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor_Gaussian(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_width: int, max_action: np.ndarray, device: torch.device):
        super(Actor_Gaussian, self).__init__()
        self.max_action = torch.FloatTensor(max_action)
        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = nn.Tanh()  # Trick10: use tanh

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

        self.to(device)

    def forward(self, s: torch.Tensor):
        '''
        Args:
            s: (batch_size, state_dim,) 状态
        Returns:
            mean: (batch_size, action_dim,) 均值
        '''
        s = s.to(self.device)
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        #max_action :(action_dim,)
        max_action = self.max_action.unsqueeze(0).repeat(s.size(0), 1) # (batch_size, action_dim,)
        mean = max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_width: int, device: torch.device):
        super(Critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, 1)
        self.activate_func = nn.tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        self.to(device)

    def forward(self, s: torch.Tensor):
        '''
        Args:
            s: (batch_size, state_dim,) 状态
        Returns:
            value: (batch_size, 1,) 价值
        '''
        s = s.to(self.device)
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        value = self.fc3(s)
        return value

class PPO_continuous(nn.Module):
    def __init__(self, config:dict):
        super(PPO_continuous, self).__init__()
        self.config = config
        self.agent_n = config.get("map", {}).get("kNumAgents", 1)
        self.goal_n = config.get("map", {}).get("kNumGoals", 1)
        self.obstacle_n = config.get("map", {}).get("kNumObstacles", 1)
        self.all_state_dim = self.agent_n * 3 + self.goal_n * 3 + self.obstacle_n * 3
        self.observe_state_dim = config.get("agent", {}).get("kNumRadars", 32)
        self.action_dim = 2
        self.max_action = np.array([config.get("agent", {}).get("kMaxSpeed", 1.0), config.get("agent", {}).get("kMaxAngularSpeed", 1.0)])
        self.device = torch.device(config.get("training", {}).get("device", "cpu"))
        self.hidden_width = config.get("network", {}).get("hidden_dim", 128)

        self.batch_size = config.get("training", {}).get("batch_size", 1024)
        self.mini_batch_size = config.get("training", {}).get("mini_batch_size", 64)
        self.K_epochs = config.get("training", {}).get("K_epochs", 10)

        self.gamma = config.get("training", {}).get("gamma", 0.99)
        self.lamda = config.get("training", {}).get("lamda", 0.95)
        self.epsilon = config.get("training", {}).get("epsilon", 0.2)
        self.entropy_coef = config.get("training", {}).get("entropy_coef", 0.01)
        self.actor_lr = config.get("training", {}).get("actor_lr", 3e-4)
        self.critic_lr = config.get("training", {}).get("critic_lr", 3e-4)

        self.actor = Actor_Gaussian(self.all_state_dim, self.observe_state_dim, self.action_dim, self.max_action, self.device)
        self.critic = Critic(self.all_state_dim, self.hidden_width, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)
        
    def choose_action(self, s: np.ndarray):
        '''
        Args:
            s: (all_state_dim,) 状态
        Returns:
            action: (action_dim,) 动作
            log_prob: (1,) 动作对数概率
        '''
        s = torch.FloatTensor(s) # (all_state_dim,)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            action = dist.sample()
            action = torch.clamp(action, -self.max_action, self.max_action)
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()
        
    
