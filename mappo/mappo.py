import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from network import Actor, Critic
from replay_buffer import ReplayBuffer
import torch.nn.functional as F


class MAPPO:
    def __init__(self, args):
        self.agent_n = args.agent_n
        self.action_dim = args.action_dim
        self.max_action = args.max_action
        self.all_state_dim = args.all_state_dim
        self.self_state_dim = args.self_state_dim
        self.observe_state_dim = args.observe_state_dim
        self.episode_limit = args.episode_limit
        
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = Actor(
            self_state_dim=self.self_state_dim,
            observe_state_dim=self.observe_state_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
            max_action=self.max_action,
            device=self.device
        )
        
        self.critic = Critic(
            all_state_dim=self.all_state_dim,
            hidden_dim=args.hidden_dim,
            device=self.device,
            agent_n=self.agent_n
        )
        
        # 创建优化器
        self.actor_optimizers = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def choose_action(self, self_state, observe_l, observe_t):
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
            
            a_n, a_logprob_n = self.actor.get_action_and_log_prob(
                self_state, observe_l, observe_t
            )
            
            return a_n.cpu().numpy(), a_logprob_n.cpu().numpy()
    
    def update(self, transition_dicts: ReplayBuffer):
        traj_len = len(transition_dicts)
        buffer = transition_dicts.get_training_data()
        states = buffer['states'] # [traj_len, all_state_dim]
        next_states = buffer['next_states'] # [traj_len, all_state_dim]
        observe_l = buffer['observe_l'] # [traj_len, agent_n, observe_state_dim]
        observe_t = buffer['observe_t'] # [traj_len, agent_n, observe_state_dim]
        actions = buffer['actions'] # [traj_len, agent_n, action_dim]
        a_logprobs = buffer['a_logprobs'] # [traj_len, agent_n, action_dim]
        rewards = buffer['rewards'] # [traj_len, agent_n]
        dones = buffer['dones'] # [traj_len]
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
            self_state = states[:, i*self.self_state_dim:(i+1)*self.self_state_dim]
            observe_l_i = observe_l[:, i, :]
            observe_t_i = observe_t[:, i, :]
            action_i = actions[:, i, :]
            # TODO:目前概率是所有动作概率和，考虑改为乘积
            old_probs_i = a_logprobs[:, i, :].sum(dim=1, keepdim=True) # [traj_len, 1]

            # 获取当前的均值和对数标准差，创建正态分布
            mean, log_std = self.actor(self_state, observe_l_i, observe_t_i)
            std = log_std.exp()
            normal_dist = torch.distributions.Normal(mean, std)
            
            # 计算当前动作的对数概率
            log_probs = normal_dist.log_prob(action_i).sum(dim=1, keepdim=True) # [traj_len, 1]
            old_log_probs = torch.log(old_probs_i).detach() # [traj_len, 1]

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages[i].unsqueeze(-1) # [traj_len, 1]
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[i].unsqueeze(-1) # [traj_len, 1]

            action_loss_i = torch.mean(-torch.min(surr1, surr2))
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

    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lamda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    
    def save_model(self, save_dir, total_steps):
        torch.save(self.actor.state_dict(), f"{save_dir}/actor_step_{int(total_steps/1000)}k.pth")
        torch.save(self.critic.state_dict(), f"{save_dir}/critic_step_{int(total_steps/1000)}k.pth")
    
    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

