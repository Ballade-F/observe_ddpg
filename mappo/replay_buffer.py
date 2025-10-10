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
    def __init__(self, agent_n: int, all_state_dim: int, observe_state_dim: int, action_dim: int, batch_size: int, step_max: int):
        """
        初始化ReplayBufferBatch，管理多条轨迹
        Args:
            agent_n: 智能体数量
            all_state_dim: 全局状态维度
            observe_state_dim: 观测维度
            action_dim: 动作维度
            batch_size: 批次大小（最多存储多少条轨迹）
            step_max: 每条轨迹的最大步数
        """
        self.agent_n = agent_n
        self.all_state_dim = all_state_dim
        self.observe_state_dim = observe_state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.step_max = step_max
        
        # 使用list存储多条轨迹，每条轨迹是一个与ReplayBuffer中buffer相同结构的字典
        self.buffers = []
        
        # 记录当前有多少条轨迹
        self.traj_count = 0
        
        # 循环覆盖的索引
        self.current_idx = 0
    
    def _get_empty_buffer(self):
        """创建一个空的buffer结构"""
        return {
            'states': [],
            'next_states': [],
            'agent_states': [],
            'observe_l': [],
            'observe_t': [],
            'actions': [],
            'a_logprobs': [],
            'rewards': [],
            'dones': [],
            'step': 0  # 记录该轨迹的当前步数
        }
    
    def store_transition(self, traj_idx: int, state, next_state, agent_state, observe_l, observe_t, action, a_logprob, reward, done):
        """
        存储指定轨迹的单步transition
        Args:
            traj_idx: 轨迹索引（从0开始）
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
        # 如果轨迹索引超过batch_size，使用循环覆盖
        actual_idx = traj_idx % self.batch_size
        
        # 如果该索引位置还没有buffer，创建新的
        if actual_idx >= len(self.buffers):
            self.buffers.append(self._get_empty_buffer())
            self.traj_count += 1
        # 如果需要覆盖旧数据
        elif traj_idx >= self.batch_size and actual_idx < len(self.buffers):
            # 重置该位置的buffer
            self.buffers[actual_idx] = self._get_empty_buffer()
        
        buffer = self.buffers[actual_idx]
        
        # 检查是否超过最大步数
        if buffer['step'] >= self.step_max:
            return  # 该轨迹已满，不再存储
        
        # 存储数据
        buffer['states'].append(state)
        buffer['next_states'].append(next_state)
        buffer['agent_states'].append(agent_state)
        buffer['observe_l'].append(observe_l)
        buffer['observe_t'].append(observe_t)
        buffer['actions'].append(action)
        buffer['a_logprobs'].append(a_logprob)
        buffer['rewards'].append(reward)
        buffer['dones'].append(done)
        buffer['step'] += 1
    
    def get_trajectory(self, traj_idx: int):
        """
        获取指定轨迹的数据
        Args:
            traj_idx: 轨迹索引
        Returns:
            dict: 包含该轨迹所有transition的字典（numpy数组格式）
        """
        actual_idx = traj_idx % self.batch_size
        if actual_idx >= len(self.buffers):
            return None
        
        buffer = self.buffers[actual_idx]
        data = {}
        for key in buffer.keys():
            if key != 'step':
                data[key] = np.array(buffer[key])
        return data
    
    def get_all_data(self):
        """
        获取所有轨迹的数据
        Returns:
            list: 包含所有轨迹数据的列表，每个元素是一个字典
        """
        all_data = []
        for buffer in self.buffers:
            data = {}
            for key in buffer.keys():
                if key != 'step':
                    data[key] = np.array(buffer[key])
            all_data.append(data)
        return all_data
    
    def get_training_data(self):
        """
        获取所有轨迹的数据，转换为tensor格式
        Returns:
            list: 包含所有轨迹数据的列表，每个元素是一个字典，值为torch.Tensor
        """
        all_data = []
        for buffer in self.buffers:
            data = {}
            for key in buffer.keys():
                if key != 'step':
                    data[key] = torch.tensor(np.array(buffer[key]), dtype=torch.float32)
            all_data.append(data)
        return all_data
    
    def reset_buffer(self):
        """重置所有buffer"""
        self.buffers = []
        self.traj_count = 0
        self.current_idx = 0
    
    def reset_trajectory(self, traj_idx: int):
        """重置指定轨迹的buffer"""
        actual_idx = traj_idx % self.batch_size
        if actual_idx < len(self.buffers):
            self.buffers[actual_idx] = self._get_empty_buffer()
    
    def get_trajectory_length(self, traj_idx: int):
        """获取指定轨迹的长度"""
        actual_idx = traj_idx % self.batch_size
        if actual_idx >= len(self.buffers):
            return 0
        return self.buffers[actual_idx]['step']
    
    def get_num_trajectories(self):
        """获取当前存储的轨迹数量"""
        return len(self.buffers)
    
    def __len__(self):
        """返回当前存储的轨迹数量"""
        return len(self.buffers)
