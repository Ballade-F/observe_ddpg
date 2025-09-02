import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from network import Actor, Critic
from data_load import ReplayBuffer, MultiAgentReplayBuffer



class AgentTeam:
    '''
        集中式训练，分布式执行
        每个agent有自己的actor，整个team共享一个critic
        每个team有一个自己的回放buffer
    '''
    def __init__(self, agent_state_dim: int, observe_dim: int, all_state_dim: int,
                 action_dim: int, num_agents: int, max_action: np.ndarray, 
                 device: torch.device, batch_size: int = 256, gamma: float = 0.9, 
                 tau: float = 0.01, lr: float = 1e-4
    ):
        #训练参数
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        #网络参数
        self.hidden_dim = 256
        self.device = device
        self.num_agents = num_agents
        self.max_action = max_action
        self.agent_state_dim = agent_state_dim
        self.observe_dim = observe_dim
        self.all_state_dim = all_state_dim
        self.all_action_dim = action_dim * num_agents
        self.action_dim = action_dim

        self.actors = [Actor(agent_state_dim, observe_dim, action_dim, self.hidden_dim, self.max_action, self.device) for _ in range(num_agents)]
        self.actor_targets = [copy.deepcopy(actor) for actor in self.actors]
        
        # Critic的action_dim应该是所有agent的动作维度总和
        self.critic = Critic(all_state_dim, self.all_action_dim, self.hidden_dim, self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    
    def choose_action(self, agent_state: np.ndarray, observe_length: np.ndarray, observe_type: np.ndarray) -> np.ndarray:
        '''
        agent_state: (num_agents, agent_state_dim) 智能体自身状态 x,y,theta
        observe_length: (num_agents, observe_dim) 雷达探测到的障碍物距离
        observe_type: (num_agents, observe_dim) 雷达探测到的障碍物类型
        '''

        actions = []
        for i in range(self.num_agents):
            agent_state_tensor = torch.FloatTensor(agent_state[i]).unsqueeze(0).to(self.device)
            observe_length_tensor = torch.FloatTensor(observe_length[i]).unsqueeze(0).to(self.device)
            observe_type_tensor = torch.FloatTensor(observe_type[i]).unsqueeze(0).to(self.device)
            action = self.actors[i](agent_state_tensor, observe_length_tensor, observe_type_tensor).cpu().data.numpy().flatten()
            actions.append(action)
        return np.array(actions)

    def learn(self, replay_buffer: MultiAgentReplayBuffer):
        if replay_buffer.size < self.batch_size:
            return
        
        # 从replay buffer中采样
        batch_states, batch_observe_lengths, batch_observe_types, batch_actions, batch_rewards, \
        batch_next_states, batch_next_observe_lengths, batch_next_observe_types, batch_dones = replay_buffer.sample(self.batch_size)
        
        # 将数据移到设备上
        batch_states = batch_states.to(self.device)  # (batch_size, all_state_dim) 全局状态
        batch_observe_lengths = batch_observe_lengths.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_observe_types = batch_observe_types.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_actions = batch_actions.to(self.device)  # (batch_size, num_agents, action_dim)
        batch_rewards = batch_rewards.to(self.device)  # (batch_size, 1)
        batch_next_states = batch_next_states.to(self.device)  # (batch_size, all_state_dim) 全局状态
        batch_next_observe_lengths = batch_next_observe_lengths.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_next_observe_types = batch_next_observe_types.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_dones = batch_dones.to(self.device)  # (batch_size, 1)

        # critic
        with torch.no_grad():
            next_actions = batch_actions.clone() # (batch_size, num_agents, action_dim)
            for idx, target_actor in enumerate(self.actor_targets):
                agent_state = batch_next_states[:, idx*self.agent_state_dim:(idx+1)*self.agent_state_dim]
                observe_length = batch_next_observe_lengths[:, idx, :]
                observe_type = batch_next_observe_types[:, idx, :]
                next_action = target_actor(agent_state, observe_length, observe_type) # (batch_size, action_dim)
                next_actions[:, idx, :] = next_action
            next_actions_concat = next_actions.view(self.batch_size, -1) # (batch_size, num_agents*action_dim)
            Q_ = self.critic_target(batch_next_states, next_actions_concat)
            target_Q = batch_rewards + self.gamma * (1 - batch_dones) * Q_
     
        batch_actions_concat = batch_actions.view(self.batch_size, -1) # (batch_size, num_agents*action_dim)
        current_Q = self.critic(batch_states, batch_actions_concat)
        critic_loss = F.mse_loss(target_Q, current_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # actor
        for params in self.critic.parameters():
                params.requires_grad = False
                
        actor_actions = batch_actions.clone()
        for idx, actor in enumerate(self.actors):
            agent_state = batch_states[:, idx*self.agent_state_dim:(idx+1)*self.agent_state_dim]
            observe_length = batch_observe_lengths[:, idx, :]
            observe_type = batch_observe_types[:, idx, :]
            actor_action = actor(agent_state, observe_length, observe_type) # (batch_size, action_dim)
            actor_actions[:, idx, :] = actor_action
            
        actor_actions_concat = actor_actions.view(self.batch_size, -1)
        actor_loss = -torch.mean(self.critic(batch_states, actor_actions_concat).flatten())
        for idx, actor_optimizer in enumerate(self.actor_optimizers):
            actor_optimizer.zero_grad()
        actor_loss.backward()
        for idx, actor_optimizer in enumerate(self.actor_optimizers):
            actor_optimizer.step()
        
        for params in self.critic.parameters():
            params.requires_grad = True

        self.update_network_parameters()
        
        
    def update_network_parameters(self):
        # critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # actor
        for idx, actor in enumerate(self.actors):
            for param, target_param in zip(actor.parameters(), self.actor_targets[idx].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


