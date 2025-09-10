import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from network import Actor, Critic
from data_load import MAPPOBuffer

class MAPPOAgent:
    """
    MAPPO算法实现 - 多智能体近端策略优化
    特点：
    1. 单个Actor网络，所有智能体共享参数
    2. 集中式Critic，使用全局状态进行价值估计
    3. 连续动作空间
    4. PPO的clip机制和价值函数损失
    """
    def __init__(self, agent_state_dim: int, observe_dim: int, all_state_dim: int,
                 action_dim: int, num_agents: int, max_action: np.ndarray, 
                 device: torch.device, batch_size: int = 256, gamma: float = 0.99, 
                 lam: float = 0.95, lr: float = 3e-4, eps_clip: float = 0.2,
                 k_epochs: int = 10, entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5):
        
        # 训练参数
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam  # GAE参数
        self.eps_clip = eps_clip  # PPO clip参数
        self.k_epochs = k_epochs  # PPO更新轮数
        self.entropy_coef = entropy_coef  # 熵系数
        self.value_coef = value_coef  # 价值函数损失系数
        self.max_grad_norm = max_grad_norm  # 梯度裁剪
        
        # 网络参数
        self.hidden_dim = 256
        self.device = device
        self.num_agents = num_agents
        self.max_action = max_action
        self.agent_state_dim = agent_state_dim
        self.observe_dim = observe_dim
        self.all_state_dim = all_state_dim
        self.action_dim = action_dim

        # 创建单个Actor网络（所有智能体共享）
        self.actor = Actor(agent_state_dim, observe_dim, action_dim, self.hidden_dim, self.max_action, self.device)
        
        # 创建集中式Critic网络
        self.critic = Critic(all_state_dim, self.hidden_dim, self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        print(f"MAPPO Agent初始化完成:")
        print(f"  Actor参数数量: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"  Critic参数数量: {sum(p.numel() for p in self.critic.parameters())}")
        print(f"  使用单个Actor网络，{num_agents}个智能体共享参数")
    
    def choose_action(self, agent_states: np.ndarray, observe_lengths: np.ndarray, 
                     observe_types: np.ndarray, deterministic: bool = False) -> tuple:
        """
        选择动作
        Args:
            agent_states: (num_agents, agent_state_dim) 智能体自身状态
            observe_lengths: (num_agents, observe_dim) 雷达探测到的障碍物距离
            observe_types: (num_agents, observe_dim) 雷达探测到的障碍物类型
            deterministic: 是否使用确定性策略
        Returns:
            actions: (num_agents, action_dim) 动作
            log_probs: (num_agents, 1) 对数概率
        """
        actions = []
        log_probs = []
        
        with torch.no_grad():
            for i in range(self.num_agents):
                agent_state_tensor = torch.FloatTensor(agent_states[i]).unsqueeze(0).to(self.device)
                observe_length_tensor = torch.FloatTensor(observe_lengths[i]).unsqueeze(0).to(self.device)
                observe_type_tensor = torch.FloatTensor(observe_types[i]).unsqueeze(0).to(self.device)
                
                action, log_prob = self.actor.get_action_and_log_prob(
                    agent_state_tensor, observe_length_tensor, observe_type_tensor, deterministic
                )
                
                actions.append(action.cpu().numpy().flatten())
                if log_prob is not None:
                    log_probs.append(log_prob.cpu().numpy().flatten())
                else:
                    log_probs.append(None)
        
        actions = np.array(actions)
        log_probs = np.array(log_probs) if log_probs[0] is not None else None
        
        return actions, log_probs
    
    def get_value(self, all_state: np.ndarray) -> float:
        """
        获取状态价值
        Args:
            all_state: (all_state_dim,) 全局状态
        Returns:
            value: 状态价值
        """
        with torch.no_grad():
            all_state_tensor = torch.FloatTensor(all_state).unsqueeze(0).to(self.device)
            value = self.critic(all_state_tensor)
            return value.cpu().numpy().flatten()[0]
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> tuple:
        """
        计算广义优势估计(GAE)
        Args:
            rewards: (episode_length,) 奖励序列
            values: (episode_length + 1,) 状态价值序列
            dones: (episode_length,) 结束标志序列
        Returns:
            advantages: (episode_length,) 优势函数
            returns: (episode_length,) 回报
        """
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self, buffer: 'MAPPOBuffer'):
        """
        使用PPO算法更新网络
        Args:
            buffer: 经验缓冲区
        """
        if buffer.size == 0:
            return {}
        
        # 获取所有数据
        states, agent_states, observe_lengths, observe_types, actions, log_probs, rewards, dones, values = buffer.get_all()
        
        # 计算优势函数和回报
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        agent_states = torch.FloatTensor(agent_states).to(self.device)  # (episode_length, num_agents, agent_state_dim)
        observe_lengths = torch.FloatTensor(observe_lengths).to(self.device)  # (episode_length, num_agents, observe_dim)
        observe_types = torch.FloatTensor(observe_types).to(self.device)  # (episode_length, num_agents, observe_dim)
        actions = torch.FloatTensor(actions).to(self.device)  # (episode_length, num_agents, action_dim)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)  # (episode_length, num_agents, 1)
        advantages = torch.FloatTensor(advantages).to(self.device)  # (episode_length,)
        returns = torch.FloatTensor(returns).to(self.device)  # (episode_length,)
        
        # 记录损失
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # PPO更新
        for epoch in range(self.k_epochs):
            # 重新计算当前策略的log_probs和values
            episode_length = states.shape[0]
            
            # 计算所有时间步和智能体的log_probs和entropy
            all_log_probs = []
            all_entropy = []
            
            for t in range(episode_length):
                step_log_probs = []
                step_entropy = []
                
                for agent_id in range(self.num_agents):
                    log_prob, entropy = self.actor.evaluate_actions(
                        agent_states[t, agent_id:agent_id+1],  # (1, agent_state_dim)
                        observe_lengths[t, agent_id:agent_id+1],  # (1, observe_dim)
                        observe_types[t, agent_id:agent_id+1],  # (1, observe_dim)
                        actions[t, agent_id:agent_id+1]  # (1, action_dim)
                    )
                    step_log_probs.append(log_prob)
                    step_entropy.append(entropy)
                
                all_log_probs.append(torch.cat(step_log_probs, dim=0))  # (num_agents, 1)
                all_entropy.append(torch.cat(step_entropy, dim=0))  # (num_agents, 1)
            
            new_log_probs = torch.stack(all_log_probs, dim=0)  # (episode_length, num_agents, 1)
            entropy = torch.stack(all_entropy, dim=0)  # (episode_length, num_agents, 1)
            
            # 计算当前价值
            current_values = self.critic(states).squeeze(-1)  # (episode_length,)
            
            # 计算ratio - 对所有智能体求平均
            ratios = torch.exp(new_log_probs - old_log_probs)  # (episode_length, num_agents, 1)
            ratios = ratios.mean(dim=1).squeeze(-1)  # (episode_length,) 对智能体维度求平均
            
            # PPO actor损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            entropy_loss = -entropy.mean()
            
            # 总的actor损失
            total_actor_loss_step = actor_loss + self.entropy_coef * entropy_loss
            
            # Critic损失
            critic_loss = F.mse_loss(current_values, returns)
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            total_actor_loss_step.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # 记录损失
            total_actor_loss += total_actor_loss_step.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        buffer.clear()
        
        # 返回训练统计
        return {
            'actor_loss': total_actor_loss / self.k_epochs,
            'critic_loss': total_critic_loss / self.k_epochs,
            'entropy': total_entropy / self.k_epochs,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'returns_mean': returns.mean().item()
        }
    
    def save_models(self, model_dir: str):
        """
        保存模型
        Args:
            model_dir: 模型保存目录
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save(self.actor.state_dict(), os.path.join(model_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(model_dir, "critic.pth"))
    
    def load_models(self, model_dir: str):
        """
        加载模型
        Args:
            model_dir: 模型目录路径
        """
        print(f"正在加载MAPPO模型从: {model_dir}")
        
        # 加载actor网络
        actor_path = os.path.join(model_dir, "actor.pth")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.actor.eval()
            print(f"已加载 Actor 模型: {actor_path}")
        else:
            raise FileNotFoundError(f"找不到Actor模型文件: {actor_path}")
        
        # 加载critic网络
        critic_path = os.path.join(model_dir, "critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic.eval()
            print(f"已加载 Critic 模型: {critic_path}")
        else:
            print(f"警告：找不到Critic模型文件: {critic_path}")
        
        print("MAPPO模型加载完成！")
