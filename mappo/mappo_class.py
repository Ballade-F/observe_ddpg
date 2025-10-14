import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from network import Actor, Critic
from replay_buffer import ReplayBuffer, ReplayBufferBatch
import torch.nn.functional as F


class MAPPO:
    def __init__(self, agent_n: int, all_state_dim: int, self_state_dim: int, 
                 observe_state_dim: int, action_dim: int, max_action: np.ndarray, 
                 gamma: float, lamda: float, epsilon: float,
                 hidden_dim: int, actor_lr: float, critic_lr: float, device: torch.device):
        """
        初始化MAPPO
        Args:
            agent_n: 智能体数量
            all_state_dim: 全局状态维度
            self_state_dim: 智能体自身状态维度
            observe_state_dim: 观测维度
            action_dim: 动作维度
            max_action: 最大动作值
            gamma: 折扣因子
            lamda: 优势函数的参数
            epsilon: 裁剪参数
            hidden_dim: 隐藏层维度
            actor_lr: actor优化器学习率
            critic_lr: critic优化器学习率
            device: 设备
        """
        self.agent_n = agent_n
        self.action_dim = action_dim
        self.max_action = max_action
        self.all_state_dim = all_state_dim
        self.self_state_dim = self_state_dim
        self.observe_state_dim = observe_state_dim
        
        # 训练参数
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        
        # 网络参数
        self.hidden_dim = hidden_dim
        
        # 设备配置
        self.device = device
        
        # 创建网络
        self.actor = Actor(
            self_state_dim=self.self_state_dim,
            observe_state_dim=self.observe_state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            max_action=self.max_action,
            device=self.device,
            all_state_dim=self.all_state_dim
        )
        
        self.critic = Critic(
            all_state_dim=self.all_state_dim,
            hidden_dim=self.hidden_dim,
            device=self.device,
            agent_n=self.agent_n
        )
        
        # 创建优化器
        self.actor_optimizers = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def choose_action(self, self_state, observe_l, observe_t, all_state):
        """
        选择动作
        Args:
            self_state: (agent_n, self_state_dim) 每个agent的自身状态
            observe_l: (agent_n, observe_state_dim) 每个agent的雷达距离观测
            observe_t: (agent_n, observe_state_dim) 每个agent的雷达类型观测
        Returns:
            a_n: (agent_n, action_dim) 动作
            a_logprob_n: (agent_n, action_dim) 动作对数概率
        """
        with torch.no_grad():
            self_state = torch.tensor(self_state, dtype=torch.float32)
            observe_l = torch.tensor(observe_l, dtype=torch.float32)
            observe_t = torch.tensor(observe_t, dtype=torch.float32)
            all_state = torch.tensor(all_state, dtype=torch.float32) #shape: (all_state_dim,)
            all_state = all_state.unsqueeze(0) #shape: (1, all_state_dim)
            
            a_n, a_logprob_n = self.actor.get_action_and_log_prob(
                self_state, observe_l, observe_t, all_state=all_state
            )
            
            return a_n.cpu().numpy(), a_logprob_n.cpu().numpy()
    
    def update(self, transition_dicts: ReplayBuffer):
        traj_len = len(transition_dicts)
        buffer = transition_dicts.get_training_data()
        agent_states = buffer['agent_states'].to(self.device) # [traj_len, agent_n, agent_state_dim]
        states = buffer['states'].to(self.device) # [traj_len, all_state_dim]
        next_states = buffer['next_states'].to(self.device) # [traj_len, all_state_dim]
        observe_l = buffer['observe_l'].to(self.device) # [traj_len, agent_n, observe_state_dim]
        observe_t = buffer['observe_t'].to(self.device) # [traj_len, agent_n, observe_state_dim]
        actions = buffer['actions'].to(self.device) # [traj_len, agent_n, action_dim]
        a_logprobs = buffer['a_logprobs'].to(self.device) # [traj_len, agent_n, action_dim]
        rewards = buffer['rewards'].to(self.device) # [traj_len, agent_n]
        dones = buffer['dones'].to(self.device) # [traj_len]
        # done复制agent_n次
        dones = dones.unsqueeze(-1).repeat(1, self.agent_n) # [traj_len, agent_n]

        # 从critic计算价值和TD-target
        values = self.critic(states) # [traj_len, agent_n]
        next_values = self.critic(next_states) # [traj_len, agent_n]
        td_target = rewards + self.gamma * next_values * (1 - dones) # [traj_len, agent_n]
        td_delta = td_target - values # [traj_len, agent_n]
        
        # 为每个智能体计算其优势
        advantages = [] # [agent_n, traj_len]
        for i in range(self.agent_n):
            adv_i = self.compute_advantage(td_delta[:, i])
            advantages.append(adv_i.to(self.device))  # [traj_len]

        #debug 先让adv=reward，维度转换一下，由[traj_len, agent_n]变为[agent_n, traj_len]
        advantages = rewards.transpose(0, 1)


        # 更新critic
        critic_loss = F.mse_loss(values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新actor
        action_losses = []
        entropies = []
        action_loss = torch.zeros(1, device=self.device)
        for i in range(self.agent_n):
            self_state = agent_states[:, i, :]
            observe_l_i = observe_l[:, i, :]
            observe_t_i = observe_t[:, i, :]
            action_i = actions[:, i, :]
            # TODO:目前概率是所有动作对数概率和，考虑改为乘积
            old_probs_i = a_logprobs[:, i, :]
            old_probs_i = old_probs_i.sum(dim=1, keepdim=False).detach() # [traj_len]

            # 获取当前的均值和对数标准差，创建正态分布
            mean, log_std = self.actor(self_state, observe_l_i, observe_t_i, states)
            std = log_std.exp()
            normal_dist = torch.distributions.Normal(mean, std)
            
            # 计算当前动作的对数概率
            log_probs = normal_dist.log_prob(action_i)
            log_probs = log_probs.sum(dim=1, keepdim=False) # [traj_len]

            ratio = torch.exp(log_probs - old_probs_i)
            surr1 = ratio * advantages[i] # [traj_len]
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[i] # [traj_len]

            # action_loss_i = torch.mean(-torch.min(surr1, surr2))
            action_loss_i = torch.mean(-surr1)
            action_loss += action_loss_i
            # 对于连续动作空间，熵的计算不同
            entropy_val = torch.mean(normal_dist.entropy()).item()

            action_losses.append(action_loss_i.item())
            entropies.append(entropy_val)

        action_loss /= torch.tensor(self.agent_n, device=self.device)
        self.actor_optimizers.zero_grad()
        action_loss.backward()
        self.actor_optimizers.step()

        return np.mean(action_losses), critic_loss.item(), np.mean(entropies)

    def update_batch(self, batch_buffer: ReplayBufferBatch):
        """
        对整个batch中的多条轨迹进行训练
        Args:
            batch_buffer: ReplayBufferBatch对象，包含多条不同长度的轨迹
        Returns:
            avg_action_loss: 平均actor损失
            avg_critic_loss: 平均critic损失
            avg_entropy: 平均熵
        """
        # 获取所有轨迹的数据
        all_trajectories = batch_buffer.get_training_data()
        num_trajs = len(all_trajectories)
        
        if num_trajs == 0:
            return 0.0, 0.0, 0.0
        
        # 初始化累积损失
        total_critic_loss = torch.zeros(1, device=self.device)
        total_action_loss = torch.zeros(1, device=self.device)
        total_action_losses_per_agent = []
        total_entropies = []
        num_valid_trajs = 0  # 记录有效轨迹数
        
        # 清零梯度
        self.critic_optimizer.zero_grad()
        self.actor_optimizers.zero_grad()
        
        # 遍历每条轨迹
        for traj_idx, buffer in enumerate(all_trajectories):
            traj_len = len(buffer['states'])
            if traj_len == 0:
                continue
            
            num_valid_trajs += 1
            
            # 提取轨迹数据
            agent_states = buffer['agent_states'].to(self.device)  # [traj_len, agent_n, agent_state_dim]
            states = buffer['states'].to(self.device)  # [traj_len, all_state_dim]
            next_states = buffer['next_states'].to(self.device)  # [traj_len, all_state_dim]
            observe_l = buffer['observe_l'].to(self.device)  # [traj_len, agent_n, observe_state_dim]
            observe_t = buffer['observe_t'].to(self.device)  # [traj_len, agent_n, observe_state_dim]
            actions = buffer['actions'].to(self.device)  # [traj_len, agent_n, action_dim]
            a_logprobs = buffer['a_logprobs'].to(self.device)  # [traj_len, agent_n, action_dim]
            rewards = buffer['rewards'].to(self.device)  # [traj_len, agent_n]
            dones = buffer['dones'].to(self.device)  # [traj_len]
            
            # done复制agent_n次
            dones = dones.unsqueeze(-1).repeat(1, self.agent_n)  # [traj_len, agent_n]
            
            # 从critic计算价值和TD-target
            values = self.critic(states)  # [traj_len, agent_n]
            next_values = self.critic(next_states)  # [traj_len, agent_n]
            td_target = rewards + self.gamma * next_values * (1 - dones)  # [traj_len, agent_n]
            td_delta = td_target - values  # [traj_len, agent_n]
            
            # 为每个智能体计算其优势
            advantages = []  # [agent_n, traj_len]
            for i in range(self.agent_n):
                adv_i = self.compute_advantage(td_delta[:, i])
                advantages.append(adv_i.to(self.device))  # [traj_len]
            
            # debug 先让adv=reward，维度转换一下，由[traj_len, agent_n]变为[agent_n, traj_len]
            advantages = rewards.transpose(0, 1)
            
            # 计算critic损失（使用mean，对该轨迹内的step和agent求平均）
            critic_loss = F.mse_loss(values, td_target.detach(), reduction='mean')
            total_critic_loss += critic_loss
            
            # 计算actor损失
            action_losses = []
            entropies = []
            
            for i in range(self.agent_n):
                self_state = agent_states[:, i, :]
                observe_l_i = observe_l[:, i, :]
                observe_t_i = observe_t[:, i, :]
                action_i = actions[:, i, :]
                old_probs_i = a_logprobs[:, i, :]
                old_probs_i = old_probs_i.sum(dim=1, keepdim=False).detach()  # [traj_len]
                
                # 获取当前的均值和对数标准差，创建正态分布
                mean, log_std = self.actor(self_state, observe_l_i, observe_t_i, states)
                std = log_std.exp()
                normal_dist = torch.distributions.Normal(mean, std)
                
                # 计算当前动作的对数概率
                log_probs = normal_dist.log_prob(action_i)
                log_probs = log_probs.sum(dim=1, keepdim=False)  # [traj_len]
                
                ratio = torch.exp(log_probs - old_probs_i)
                surr1 = ratio * advantages[i]  # [traj_len]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[i]
                
                # 对该轨迹的该智能体的所有步求平均
                # action_loss_i = torch.mean(-torch.min(surr1, surr2))
                action_loss_i = torch.mean(-surr1)
                total_action_loss += action_loss_i
                
                # 对于连续动作空间，熵的计算
                entropy_val = torch.mean(normal_dist.entropy()).item()
                
                action_losses.append(action_loss_i.item())
                entropies.append(entropy_val)
            
            total_action_losses_per_agent.append(action_losses)
            total_entropies.append(entropies)
        
        # 如果没有有效轨迹，返回0
        if num_valid_trajs == 0:
            return 0.0, 0.0, 0.0
        
        # 计算平均损失（对所有轨迹求平均）
        avg_critic_loss = total_critic_loss / num_valid_trajs
        avg_action_loss = total_action_loss / (num_valid_trajs * self.agent_n)
        
        # 反向传播并更新
        avg_critic_loss.backward()
        self.critic_optimizer.step()
        
        avg_action_loss.backward()
        self.actor_optimizers.step()
        
        # 计算统计信息
        all_action_losses = []
        all_entropies = []
        for traj_losses in total_action_losses_per_agent:
            all_action_losses.extend(traj_losses)
        for traj_entropies in total_entropies:
            all_entropies.extend(traj_entropies)
        
        return np.mean(all_action_losses) if all_action_losses else 0.0, \
               avg_critic_loss.item(), \
               np.mean(all_entropies) if all_entropies else 0.0

    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lamda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    
    def save_model(self, save_dir, episode):
        torch.save(self.actor.state_dict(), f"{save_dir}/actor_episode_{episode/100}.pth")
        torch.save(self.critic.state_dict(), f"{save_dir}/critic_episode_{episode/100}.pth")
    
    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

