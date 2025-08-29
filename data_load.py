#数据回放，存储，加载，(归一化)
#应当包含sim_env


#TODO: 多个已经随机生成好的sim_env，加载config和数据。

import numpy as np
from typing import Tuple
import torch

class ReplayBuffer:
    def __init__(self, state_dim: int, observe_dim: int, action_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.observe_length = np.zeros((max_size, observe_dim))
        self.observe_type = np.zeros((max_size, observe_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_observe_length = np.zeros((max_size, observe_dim))
        self.next_observe_type = np.zeros((max_size, observe_dim))
        self.done = np.zeros((max_size, 1))

    def store(self, state: np.ndarray, observe_length: np.ndarray, observe_type: np.ndarray, 
              action: np.ndarray, reward: float, 
              next_state: np.ndarray, next_observe_length: np.ndarray, next_observe_type: np.ndarray, done: bool):
        self.state[self.count] = state
        self.observe_length[self.count] = observe_length
        self.observe_type[self.count] = observe_type
        self.action[self.count] = action
        self.reward[self.count] = reward
        self.next_state[self.count] = next_state
        self.next_observe_length[self.count] = next_observe_length
        self.next_observe_type[self.count] = next_observe_type
        self.done[self.count] = done

        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size: int) :
        '''
        return:
        state: (batch_size, state_dim)
        observe_length: (batch_size, observe_dim)
        observe_type: (batch_size, observe_dim)
        action: (batch_size, action_dim)
        reward: (batch_size, 1)
        next_state: (batch_size, state_dim)
        next_observe_length: (batch_size, observe_dim)
        next_observe_type: (batch_size, observe_dim)
        done: (batch_size, 1)
        '''
        index = np.random.choice(self.size, size=batch_size)
        batch_state = torch.FloatTensor(self.state[index])
        batch_observe_length = torch.FloatTensor(self.observe_length[index])
        batch_observe_type = torch.FloatTensor(self.observe_type[index])
        batch_action = torch.FloatTensor(self.action[index])
        batch_reward = torch.FloatTensor(self.reward[index])
        batch_next_state = torch.FloatTensor(self.next_state[index])
        batch_next_observe_length = torch.FloatTensor(self.next_observe_length[index])
        batch_next_observe_type = torch.FloatTensor(self.next_observe_type[index])
        batch_done = torch.FloatTensor(self.done[index])

        return (batch_state, batch_observe_length, batch_observe_type, batch_action, batch_reward, \
            batch_next_state, batch_next_observe_length, batch_next_observe_type, batch_done)
    

class MultiAgentReplayBuffer:
    def __init__(self, state_dim: int, observe_dim: int, action_dim: int, num_agents: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.num_agents = num_agents
        
        # 存储全局状态和各agent的观测、动作
        # state_dim: 全局状态维度 = agent_state_dim*num_agents + goal_state_dim*num_goals + obstacles_state_dim*num_obstacles
        # 即 3*num_agents + 3*num_goals + 3*num_obstacles
        self.states = np.zeros((max_size, state_dim))  # 全局状态
        self.observe_lengths = np.zeros((max_size, num_agents, observe_dim))  # 每个agent的观测长度
        self.observe_types = np.zeros((max_size, num_agents, observe_dim))  # 每个agent的观测类型
        self.actions = np.zeros((max_size, num_agents, action_dim))  # 每个agent的动作
        self.rewards = np.zeros((max_size, num_agents))  # 每个agent的奖励
        self.next_states = np.zeros((max_size, state_dim))  # 下一步全局状态
        self.next_observe_lengths = np.zeros((max_size, num_agents, observe_dim))  # 下一步每个agent的观测长度
        self.next_observe_types = np.zeros((max_size, num_agents, observe_dim))  # 下一步每个agent的观测类型
        self.dones = np.zeros((max_size, 1))  # 环境结束标志，所有agent共享

    def store(self, state: np.ndarray, observe_lengths: np.ndarray, observe_types: np.ndarray, 
              actions: np.ndarray, rewards: np.ndarray, 
              next_state: np.ndarray, next_observe_lengths: np.ndarray, next_observe_types: np.ndarray, done: int):
        """
        存储多个agent的经验
        Args:
            state: (state_dim,) 全局状态
            observe_lengths: (num_agents, observe_dim) 每个agent的观测长度
            observe_types: (num_agents, observe_dim) 每个agent的观测类型
            actions: (num_agents, action_dim) 每个agent的动作
            rewards: (num_agents,) 每个agent的奖励
            next_state: (state_dim,) 下一步全局状态
            next_observe_lengths: (num_agents, observe_dim) 下一步每个agent的观测长度
            next_observe_types: (num_agents, observe_dim) 下一步每个agent的观测类型
            done: int 环境结束标志
        """
        self.states[self.count] = state
        self.observe_lengths[self.count] = observe_lengths
        self.observe_types[self.count] = observe_types
        self.actions[self.count] = actions
        self.rewards[self.count] = rewards
        self.next_states[self.count] = next_state
        self.next_observe_lengths[self.count] = next_observe_lengths
        self.next_observe_types[self.count] = next_observe_types
        self.dones[self.count] = done

        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size: int):
        """
        采样批量数据
        Returns:
            states: (batch_size, state_dim) 全局状态
            observe_lengths: (batch_size, num_agents, observe_dim) 每个agent的观测长度
            observe_types: (batch_size, num_agents, observe_dim) 每个agent的观测类型
            actions: (batch_size, num_agents, action_dim) 每个agent的动作
            rewards: (batch_size, num_agents) 每个agent的奖励
            next_states: (batch_size, state_dim) 下一步全局状态
            next_observe_lengths: (batch_size, num_agents, observe_dim) 下一步每个agent的观测长度
            next_observe_types: (batch_size, num_agents, observe_dim) 下一步每个agent的观测类型
            dones: (batch_size, 1) 环境结束标志
        """
        index = np.random.choice(self.size, size=batch_size)
        batch_states = torch.FloatTensor(self.states[index])
        batch_observe_lengths = torch.FloatTensor(self.observe_lengths[index])
        batch_observe_types = torch.FloatTensor(self.observe_types[index])
        batch_actions = torch.FloatTensor(self.actions[index])
        batch_rewards = torch.FloatTensor(self.rewards[index])
        batch_next_states = torch.FloatTensor(self.next_states[index])
        batch_next_observe_lengths = torch.FloatTensor(self.next_observe_lengths[index])
        batch_next_observe_types = torch.FloatTensor(self.next_observe_types[index])
        batch_dones = torch.FloatTensor(self.dones[index])

        return (batch_states, batch_observe_lengths, batch_observe_types, batch_actions, batch_rewards,
                batch_next_states, batch_next_observe_lengths, batch_next_observe_types, batch_dones)
    

    
