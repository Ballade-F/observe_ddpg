# MAPPO数据缓冲区实现
# 与MADDPG不同，MAPPO使用on-policy学习，需要存储轨迹数据而不是经验回放

import numpy as np
from typing import Tuple
import torch

class MAPPOBuffer:
    """
    MAPPO算法的轨迹缓冲区
    特点：
    1. 存储完整的episode轨迹
    2. 包含状态价值估计
    3. 支持GAE计算
    4. 用完即清空（on-policy）
    """
    def __init__(self, max_episode_length: int = 1000):
        self.max_episode_length = max_episode_length
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.states = []  # 全局状态
        self.agent_states = []  # 各智能体状态
        self.observe_lengths = []  # 各智能体观测长度
        self.observe_types = []  # 各智能体观测类型
        self.actions = []  # 各智能体动作
        self.log_probs = []  # 动作的对数概率
        self.rewards = []  # 奖励
        self.dones = []  # 结束标志
        self.values = []  # 状态价值
        self.size = 0
    
    def store(self, state: np.ndarray, agent_states: np.ndarray, observe_lengths: np.ndarray, 
              observe_types: np.ndarray, actions: np.ndarray, log_probs: np.ndarray, 
              reward: float, done: bool, value: float):
        """
        存储一个时间步的数据
        Args:
            state: (all_state_dim,) 全局状态
            agent_states: (num_agents, agent_state_dim) 各智能体状态
            observe_lengths: (num_agents, observe_dim) 各智能体观测长度
            observe_types: (num_agents, observe_dim) 各智能体观测类型
            actions: (num_agents, action_dim) 各智能体动作
            log_probs: (num_agents, 1) 动作的对数概率
            reward: float 奖励
            done: bool 结束标志
            value: float 状态价值
        """
        if self.size >= self.max_episode_length:
            print(f"警告：缓冲区已满，忽略新数据")
            return
        
        self.states.append(state.copy())
        self.agent_states.append(agent_states.copy())
        self.observe_lengths.append(observe_lengths.copy())
        self.observe_types.append(observe_types.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.size += 1
    
    def store_final_value(self, final_value: float):
        """
        存储最终状态的价值估计（用于GAE计算）
        Args:
            final_value: float 最终状态的价值
        """
        self.values.append(final_value)
    
    def get_all(self) -> Tuple[np.ndarray, ...]:
        """
        获取所有存储的数据
        Returns:
            states: (episode_length, all_state_dim)
            agent_states: (episode_length, num_agents, agent_state_dim)
            observe_lengths: (episode_length, num_agents, observe_dim)
            observe_types: (episode_length, num_agents, observe_dim)
            actions: (episode_length, num_agents, action_dim)
            log_probs: (episode_length, num_agents, 1)
            rewards: (episode_length,)
            dones: (episode_length,)
            values: (episode_length + 1,)  # 包含最终状态价值
        """
        if self.size == 0:
            raise ValueError("缓冲区为空")
        
        states = np.array(self.states)
        agent_states = np.array(self.agent_states)
        observe_lengths = np.array(self.observe_lengths)
        observe_types = np.array(self.observe_types)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        
        return states, agent_states, observe_lengths, observe_types, actions, log_probs, rewards, dones, values

class MultiEnvMAPPOBuffer:
    """
    多环境MAPPO缓冲区
    管理多个环境的轨迹数据
    """
    def __init__(self, num_envs: int, max_episode_length: int = 1000):
        self.num_envs = num_envs
        self.buffers = [MAPPOBuffer(max_episode_length) for _ in range(num_envs)]
        self.current_env = 0
    
    def store(self, env_id: int, state: np.ndarray, agent_states: np.ndarray, 
              observe_lengths: np.ndarray, observe_types: np.ndarray, actions: np.ndarray, 
              log_probs: np.ndarray, reward: float, done: bool, value: float):
        """存储指定环境的数据"""
        self.buffers[env_id].store(state, agent_states, observe_lengths, observe_types, 
                                  actions, log_probs, reward, done, value)
    
    def store_final_value(self, env_id: int, final_value: float):
        """存储指定环境的最终价值"""
        self.buffers[env_id].store_final_value(final_value)
    
    def get_ready_buffers(self) -> list:
        """获取有数据的缓冲区"""
        ready_buffers = []
        for i, buffer in enumerate(self.buffers):
            if buffer.size > 0:
                ready_buffers.append((i, buffer))
        return ready_buffers
    
    def clear_buffer(self, env_id: int):
        """清空指定环境的缓冲区"""
        self.buffers[env_id].clear()
    
    def clear_all(self):
        """清空所有缓冲区"""
        for buffer in self.buffers:
            buffer.clear()
    
    def get_total_samples(self) -> int:
        """获取所有缓冲区的样本总数"""
        return sum(buffer.size for buffer in self.buffers)

# 为了兼容性，也提供原来的MultiAgentReplayBuffer（虽然MAPPO不使用）
class MultiAgentReplayBuffer:
    def __init__(self, state_dim: int, observe_dim: int, action_dim: int, num_agents: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.num_agents = num_agents
        
        # 存储全局状态和各agent的观测、动作
        self.states = np.zeros((max_size, state_dim))  # 全局状态
        self.observe_lengths = np.zeros((max_size, num_agents, observe_dim))  # 每个agent的观测长度
        self.observe_types = np.zeros((max_size, num_agents, observe_dim))  # 每个agent的观测类型
        self.actions = np.zeros((max_size, num_agents, action_dim))  # 每个agent的动作
        self.rewards = np.zeros((max_size, 1))  # 整个team的奖励
        self.next_states = np.zeros((max_size, state_dim))  # 下一步全局状态
        self.next_observe_lengths = np.zeros((max_size, num_agents, observe_dim))  # 下一步每个agent的观测长度
        self.next_observe_types = np.zeros((max_size, num_agents, observe_dim))  # 下一步每个agent的观测类型
        self.dones = np.zeros((max_size, 1))  # 环境结束标志，所有agent共享

    def store(self, state: np.ndarray, observe_lengths: np.ndarray, observe_types: np.ndarray, 
              actions: np.ndarray, rewards: float, 
              next_state: np.ndarray, next_observe_lengths: np.ndarray, next_observe_types: np.ndarray, done: int):
        """
        存储多个agent的经验
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
