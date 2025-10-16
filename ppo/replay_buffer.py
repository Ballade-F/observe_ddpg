import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, agent_n: int, all_state_dim: int, observe_state_dim: int, action_dim: int, batch_size: int):
        """
        初始化ReplayBuffer
        Args:
            agent_n: 智能体数量
            all_state_dim: 全局状态维度
            observe_state_dim: 观测维度
            action_dim: 动作维度
            batch_size: 批次大小
        """
        self.agent_n = agent_n
        self.all_state_dim = all_state_dim
        self.observe_state_dim = observe_state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.count = 0
        self.reset_buffer()

    def reset_buffer(self):
        self.states = np.zeros((self.batch_size, self.all_state_dim))
        self.actions = np.zeros((self.batch_size, self.agent_n, self.action_dim))
        self.a_logprobs = np.zeros((self.batch_size, self.agent_n, self.action_dim))
        self.rewards = np.zeros((self.batch_size, self.agent_n, 1))
        self.next_states = np.zeros((self.batch_size, self.all_state_dim))
        self.observe_l = np.zeros((self.batch_size, self.agent_n, self.observe_state_dim))
        self.observe_t = np.zeros((self.batch_size, self.agent_n, self.observe_state_dim))
        self.dones = np.zeros((self.batch_size, 1))
        self.count = 0

    def store_transition(self, state, action, a_logprob, reward, next_state, observe_l, observe_t, done):
        """
        存储单步transition
        Args:
            state: (all_state_dim,) 全局状态
            action: (agent_n, action_dim) 动作
            a_logprob: (agent_n, action_dim) 动作对数概率
            reward: (agent_n, 1) 奖励
            next_state: (all_state_dim,) 下一个全局状态
            observe_l: (agent_n, observe_state_dim) 雷达距离观测
            observe_t: (agent_n, observe_state_dim) 雷达类型观测
            done: bool 是否结束
        """
        self.states[self.count] = state
        self.actions[self.count] = action
        self.a_logprobs[self.count] = a_logprob
        self.rewards[self.count] = reward
        self.next_states[self.count] = next_state
        self.observe_l[self.count] = observe_l
        self.observe_t[self.count] = observe_t
        self.dones[self.count] = done
        self.count = (self.count + 1) % self.batch_size

    def get_data(self):
        """
        获取buffer中的数据，转换为tensor数组
        Returns:
            dict: 包含所有transition的字典,值为torch.Tensor
        """
        data = {}
        data['states'] = torch.tensor(self.states, dtype=torch.float32)
        data['actions'] = torch.tensor(self.actions, dtype=torch.float32)
        data['a_logprobs'] = torch.tensor(self.a_logprobs, dtype=torch.float32)
        data['rewards'] = torch.tensor(self.rewards, dtype=torch.float32)
        data['next_states'] = torch.tensor(self.next_states, dtype=torch.float32)
        data['observe_l'] = torch.tensor(self.observe_l, dtype=torch.float32)
        data['observe_t'] = torch.tensor(self.observe_t, dtype=torch.float32)
        data['dones'] = torch.tensor(self.dones, dtype=torch.float32)
        return data