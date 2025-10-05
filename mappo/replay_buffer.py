import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, agent_n: int, all_state_dim: int, observe_state_dim: int, action_dim: int, step_max: int):
        """
        初始化ReplayBuffer
        Args:
            agent_n: 智能体数量
            all_state_dim: 全局状态维度
            observe_state_dim: 观测维度
            action_dim: 动作维度
            step_max: 最大步数
        """
        self.agent_n = agent_n
        self.all_state_dim = all_state_dim
        self.observe_state_dim = observe_state_dim
        self.action_dim = action_dim
        self.step_max = step_max
        self.step = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'states': [],
            'next_states': [],
            'agent_states': [],
            'observe_l': [],
            'observe_t': [],
            'actions': [],
            'a_logprobs': [],
            'rewards': [],
            'dones': []
        }
        self.step = 0
    
    def store_transition(self, state, next_state, agent_state, observe_l, observe_t, action, a_logprob, reward, done):
        """
        存储单步transition
        Args:
            state: (all_state_dim,) 全局状态
            next_state: (all_state_dim,) 下一个全局状态
            agent_state: (agent_n, agent_state_dim) 每个agent的自身状态
            observe_l: (agent_n, observe_state_dim) 雷达距离观测
            observe_t: (agent_n, observe_state_dim) 雷达类型观测
            action: (agent_n, action_dim) 动作
            a_logprob: (agent_n, action_dim) 动作对数概率
            reward: (agent_n,) 每个智能体的奖励
            done: bool 是否结束
        """
        if self.step >= self.step_max:
            return  # 达到最大步数，不再存储
        
        self.buffer['states'].append(state)
        self.buffer['next_states'].append(next_state)
        self.buffer['agent_states'].append(agent_state)
        self.buffer['observe_l'].append(observe_l)
        self.buffer['observe_t'].append(observe_t)
        self.buffer['actions'].append(action)
        self.buffer['a_logprobs'].append(a_logprob)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.step += 1
    
    def get_data(self):
        """
        获取buffer中的数据，转换为numpy数组
        Returns:
            dict: 包含所有transition的字典
        """
        data = {}
        for key in self.buffer.keys():
            data[key] = np.array(self.buffer[key])
        return data
    
    def get_training_data(self):
        """
        获取buffer中的数据，转换为tensor
        Returns:
            dict: 包含所有transition的字典，值为torch.Tensor
        """
        data = {}
        for key in self.buffer.keys():
            data[key] = torch.tensor(np.array(self.buffer[key]), dtype=torch.float32)
        return data
    
    def is_full(self):
        """检查buffer是否已满"""
        return self.step >= self.step_max
    
    def __len__(self):
        return self.step
        

class ReplayBufferBatch:
    def __init__(self, agent_n: int, all_state_dim: int, agent_state_dim: int, observe_state_dim: int, action_dim: int, episode_limit: int, batch_size: int):
        """
        初始化ReplayBufferBatch
        Args:
            agent_n: 智能体数量
            all_state_dim: 全局状态维度
            agent_state_dim: 每个agent的自身状态维度
            observe_state_dim: 观测维度
            action_dim: 动作维度
            episode_limit: 每个episode的最大步数
            batch_size: 每个batch的大小
        """
        self.agent_n = agent_n
        self.all_state_dim = all_state_dim
        self.agent_state_dim = agent_state_dim
        self.observe_state_dim = observe_state_dim
        self.action_dim = action_dim
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'states': np.empty([self.batch_size, self.episode_limit, self.all_state_dim]),
            'next_states': np.empty([self.batch_size, self.episode_limit, self.all_state_dim]),
            'agent_states': np.empty([self.batch_size, self.episode_limit, self.agent_n, self.agent_state_dim]),
            'observe_l': np.empty([self.batch_size, self.episode_limit, self.agent_n, self.observe_state_dim]),
            'observe_t': np.empty([self.batch_size, self.episode_limit, self.agent_n, self.observe_state_dim]),
            'actions': np.empty([self.batch_size, self.episode_limit, self.agent_n, self.action_dim]),
            'a_logprobs': np.empty([self.batch_size, self.episode_limit, self.agent_n, self.action_dim]),
            'values': np.empty([self.batch_size, self.episode_limit+1]),
            'rewards': np.empty([self.batch_size, self.episode_limit, self.agent_n]),
            'dones': np.empty([self.batch_size, self.episode_limit])
        }
        self.episode_num = 0

    def store_transition(self, episode_step, states, next_states, agent_states, observe_l, observe_t, actions, a_logprobs, rewards, dones):
        """
        存储单步transition
        Args:
            episode_step: 当前episode的步数
            states: (all_state_dim,) 全局状态
            next_states: (all_state_dim,) 下一个全局状态
            agent_states: (agent_n, agent_state_dim) 每个agent的自身状态
            observe_l: (agent_n, observe_state_dim) 雷达距离观测
            observe_t: (agent_n, observe_state_dim) 雷达类型观测
            actions: (agent_n, action_dim) 动作
            a_logprobs: (agent_n, action_dim) 动作对数概率
            rewards: (agent_n,) 每个智能体的奖励
            dones: bool 是否结束
        """
        self.buffer['states'][self.episode_num][episode_step] = states
        self.buffer['next_states'][self.episode_num][episode_step] = next_states
        self.buffer['agent_states'][self.episode_num][episode_step] = agent_states
        self.buffer['observe_l'][self.episode_num][episode_step] = observe_l
        self.buffer['observe_t'][self.episode_num][episode_step] = observe_t
        self.buffer['actions'][self.episode_num][episode_step] = actions
        self.buffer['a_logprobs'][self.episode_num][episode_step] = a_logprobs
        self.buffer['rewards'][self.episode_num][episode_step] = rewards
        self.buffer['dones'][self.episode_num][episode_step] = dones

    def store_last_value(self, episode_step, v_n):
        self.buffer['values'][self.episode_num][episode_step] = v_n
        self.episode_num += 1


    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch

