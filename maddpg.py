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
                 lr: float, replay_buffer: MultiAgentReplayBuffer, device: torch.device
    ):
        self.hidden_dim = 256
        self.tau = 0.01 #软更新参数
        self.lr = lr
        self.device = device
        self.num_agents = num_agents
        self.max_action = max_action
        self.agent_state_dim = agent_state_dim
        self.observe_dim = observe_dim
        self.all_state_dim = all_state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer

        self.actors = [Actor(agent_state_dim, observe_dim, action_dim, self.hidden_dim, self.max_action, self.device) for _ in range(num_agents)]
        self.actor_targets = [copy.deepcopy(actor) for actor in self.actors]
        
        # Critic的action_dim应该是所有agent的动作维度总和
        total_action_dim = action_dim * num_agents
        self.critic = Critic(all_state_dim, total_action_dim, self.hidden_dim, self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.MseLoss = nn.MSELoss()


class MADDPG:
    def __init__(self, state_dim: int, observe_dim: int, action_dim: int, 
        all_state_dim: int, num_agents: int, max_action: np.ndarray, device: torch.device
    ):
        self.hidden_width = 256
        self.batch_size = 256
        self.GAMMA = 0.9
        self.TAU = 0.01
        self.lr = 1e-4

        self.num_agents = num_agents
        self.max_action = max_action
        self.state_dim = state_dim
        self.observe_dim = observe_dim
        self.action_dim = action_dim
        self.device = device

        self.all_state_dim = all_state_dim

        self.agents = [
            AgentTeam(
                self.state_dim, self.observe_dim, self.all_state_dim, self.action_dim, self.num_agents, self.max_action, self.lr, 
                self.device
                ) for _ in range(self.num_agents)]
        
        self.loss_fn = nn.MSELoss()

    def choose_action(self, agent_states: np.ndarray, observe_length: np.ndarray, observe_type: np.ndarray) -> np.ndarray:
        actions = []
        for idx, agent in enumerate(self.agents):
            agent_state = torch.FloatTensor(agent_states[idx]).unsqueeze(0).to(self.device)
            observe_length = torch.FloatTensor(observe_length[idx]).unsqueeze(0).to(self.device)
            observe_type = torch.FloatTensor(observe_type[idx]).unsqueeze(0).to(self.device)
            action = agent.actor(agent_state, observe_length, observe_type).cpu().data.numpy().flatten()
            actions.append(action)
        return np.array(actions)
        
    
    def learn(self, replay_buffer: MultiAgentReplayBuffer):
        # 检查buffer中是否有足够的经验
        if replay_buffer.size < self.batch_size:
            return
            
        # 从replay buffer中采样
        batch_states, batch_observe_lengths, batch_observe_types, batch_actions, batch_rewards, \
        batch_next_states, batch_next_observe_lengths, batch_next_observe_types, batch_dones = replay_buffer.sample(self.batch_size)
        
        # 将数据移到设备上
        batch_states = batch_states.to(self.device)  # (batch_size, state_dim) 全局状态
        batch_observe_lengths = batch_observe_lengths.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_observe_types = batch_observe_types.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_actions = batch_actions.to(self.device)  # (batch_size, num_agents, action_dim)
        batch_rewards = batch_rewards.to(self.device)  # (batch_size, num_agents)
        batch_next_states = batch_next_states.to(self.device)  # (batch_size, state_dim) 全局状态
        batch_next_observe_lengths = batch_next_observe_lengths.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_next_observe_types = batch_next_observe_types.to(self.device)  # (batch_size, num_agents, observe_dim)
        batch_dones = batch_dones.to(self.device)  # (batch_size, 1)
        
        # 对每个agent进行训练
        for agent_idx in range(self.num_agents):
            agent = self.agents[agent_idx]
            
            # 获取当前agent的观测数据
            agent_observe_lengths = batch_observe_lengths[:, agent_idx, :]  # (batch_size, observe_dim)
            agent_observe_types = batch_observe_types[:, agent_idx, :]  # (batch_size, observe_dim)
            agent_actions = batch_actions[:, agent_idx, :]  # (batch_size, action_dim)
            agent_rewards = batch_rewards[:, agent_idx].unsqueeze(1)  # (batch_size, 1)
            agent_next_observe_lengths = batch_next_observe_lengths[:, agent_idx, :]  # (batch_size, observe_dim)
            agent_next_observe_types = batch_next_observe_types[:, agent_idx, :]  # (batch_size, observe_dim)
            
            # 从全局状态中提取当前agent的状态
            # 假设全局状态结构：[agent1_state, agent2_state, ..., goals, obstacles]
            agent_state_start = agent_idx * self.state_dim  # 每个agent状态是3维(x,y,theta)
            agent_state_end = (agent_idx + 1) * self.state_dim
            agent_states = batch_states[:, agent_state_start:agent_state_end]  # (batch_size, 3)
            agent_next_states = batch_next_states[:, agent_state_start:agent_state_end]  # (batch_size, 3)
            
            # 计算目标Q值
            with torch.no_grad():
                # 使用目标actor网络计算下一状态的动作
                next_actions = []
                for i in range(self.num_agents):
                    # 从全局状态中提取每个agent的状态
                    i_state_start = i * 3
                    i_state_end = (i + 1) * 3
                    i_next_state = batch_next_states[:, i_state_start:i_state_end]
                    i_next_observe_length = batch_next_observe_lengths[:, i, :]
                    i_next_observe_type = batch_next_observe_types[:, i, :]
                    next_action = self.agents[i].actor_target(i_next_state, i_next_observe_length, i_next_observe_type)
                    next_actions.append(next_action) # next_action:(batch_size, action_dim)
                
                # 将所有agent的动作连接起来作为critic的输入
                next_actions_concat = torch.cat(next_actions, dim=1).to(self.device) # (batch_size, num_agents*action_dim)
                
                # 计算目标Q值 - 使用全局状态
                Q_ = agent.critic_target(batch_next_states, next_actions_concat)
                target_Q = agent_rewards + self.GAMMA * (1 - batch_dones) * Q_
            
            # 计算当前Q值
            current_actions_concat = batch_actions.clone()
            current_actions_concat = current_actions_concat.view(self.batch_size, -1)  # (batch_size, num_agents*action_dim)
            current_Q = agent.critic(batch_states, current_actions_concat)
            # 计算critic损失，反向传播
            critic_loss = agent.MseLoss(target_Q, current_Q)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # 更新Actor
            # 冻结critic参数以避免不必要的梯度计算
            for params in agent.critic.parameters():
                params.requires_grad = False
            
            # 计算actor损失
            actor_actions = batch_actions.clone()
            actor_actions = actor_actions.view(self.batch_size, -1)
            self_action = agent.actor(agent_states, agent_observe_lengths, agent_observe_types)
            actor_actions[:, agent_idx*self.action_dim:(agent_idx+1)*self.action_dim] = self_action
            actor_loss = -agent.critic(batch_states, actor_actions).mean()
            # actor_actions_list = []
            # for i in range(self.num_agents):
            #     if i == agent_idx:
            #         self_action = agent.actor(agent_states, agent_observe_lengths, agent_observe_types)
            #         actor_actions_list.append(self_action)
            #     else:
            #         actor_actions_list.append(batch_actions[:, i,:].detach())
            # actor_actions_concat = torch.cat(actor_actions_list, dim=1)
            # actor_loss = -agent.critic(batch_states, actor_actions_concat).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # 解冻critic参数
            for params in agent.critic.parameters():
                params.requires_grad = True
            
            # 软更新目标网络
            for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)
            
            for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)

    def build_global_state(self, agent_states: np.ndarray, goal_states: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """
        构建全局状态
        Args:
            agent_states: (num_agents, 3) 所有agent的状态 [x, y, theta]
            goal_states: (num_goals, 3) 所有目标的状态 [x, y, finish_flag]
            obstacles: (num_obstacles, 3) 所有障碍物的状态 [x, y, r]
        Returns:
            global_state: (all_state_dim,) 全局状态
        """
        # 将所有状态连接成一个全局状态向量
        global_state = np.concatenate([
            agent_states.flatten(),  # 所有agent状态
            goal_states.flatten(),   # 所有目标状态
            obstacles.flatten()      # 所有障碍物状态
        ])
        return global_state
