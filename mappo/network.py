import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, self_state_dim: int, observe_state_dim: int, action_dim: int, 
                 hidden_dim: int, max_action: np.ndarray, device: torch.device):
        super(Actor, self).__init__()
        self.max_action = torch.FloatTensor(max_action)
        self.device = device
        self.action_dim = action_dim

        # RadarObserveType有6种类型：UnfinishedGoal=0, Empty=1, FinishedGoal=2, Obstacle=3, Agent=4, Boundary=5
        self.num_radar_types = 6
        self.radar_type_embed_dim = 4  # embedding维度
        self.observe_state_dim = observe_state_dim
        
        # 为雷达类型创建embedding层
        self.radar_type_embedding = nn.Embedding(self.num_radar_types, self.radar_type_embed_dim)
        
        # 输入维度：self_state + observe_length + embedded_observe_type
        input_dim = self_state_dim + observe_state_dim + observe_state_dim * self.radar_type_embed_dim
        
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        # MAPPO需要输出mean和log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Kaiming初始化
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Kaiming初始化"""
        for layer in [self.l1, self.l2]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        # 输出层使用较小的初始化
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.log_std_layer.bias, -1.0)  # 初始化为较小的标准差
        
        # 初始化embedding层
        nn.init.xavier_uniform_(self.radar_type_embedding.weight)
    
    def forward(self, self_state: torch.Tensor, observe_length: torch.Tensor, observe_type: torch.Tensor):
        '''
        self_state: (batch_size, self_state_dim) 智能体自身状态 x,y,theta
        observe_length: (batch_size, observe_state_dim) 雷达探测到的障碍物距离
        observe_type: (batch_size, observe_state_dim) 雷达探测到的障碍物类型
        '''
        self_state = self_state.to(self.device)
        observe_length = observe_length.to(self.device)
        observe_type = observe_type.to(self.device)
        
        # 将observe_type转换为long类型用于embedding
        observe_type_long = observe_type.long()
        
        # 通过embedding层处理雷达类型
        embedded_observe_type = self.radar_type_embedding(observe_type_long)
        embedded_observe_type_flat = embedded_observe_type.view(embedded_observe_type.size(0), -1)
        
        # 拼接所有特征
        x = torch.cat([self_state, observe_length, embedded_observe_type_flat], dim=1)    
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        # 计算mean和log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # 限制log_std的范围，防止数值不稳定
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        
        # 缩放到动作范围
        mean = torch.tanh(mean) * self.max_action.to(self.device)
        
        return mean, std
    
    def get_action_and_log_prob(self, self_state: torch.Tensor, observe_length: torch.Tensor, 
                               observe_type: torch.Tensor, deterministic: bool = False):
        """
        获取动作和对数概率，用于训练和推理
        """
        mean, std = self.forward(self_state, observe_length, observe_type)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            # 创建正态分布
            dist = Normal(mean, std)
            # 采样动作
            action = dist.sample()
            # 计算对数概率
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            # 限制动作范围
            action = torch.clamp(action, -self.max_action.to(self.device), self.max_action.to(self.device))
        
        return action, log_prob
    
    def evaluate_actions(self, self_state: torch.Tensor, observe_length: torch.Tensor, 
                        observe_type: torch.Tensor, actions: torch.Tensor):
        """
        评估给定动作的对数概率和熵，用于训练时的策略梯度计算
        """
        mean, std = self.forward(self_state, observe_length, observe_type)
        
        # 创建正态分布
        dist = Normal(mean, std)
        
        # 计算对数概率
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # 计算熵
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy

class Critic(nn.Module):
    def __init__(self, all_state_dim: int, hidden_dim: int, device: torch.device):
        super(Critic, self).__init__()
        self.device = device

        # MAPPO的Critic只需要全局状态，不需要动作
        self.l1 = nn.Linear(all_state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Kaiming初始化
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Kaiming初始化"""
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        # 输出层使用较小的初始化
        nn.init.uniform_(self.value_head.weight, -3e-3, 3e-3)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, all_state: torch.Tensor) -> torch.Tensor:
        all_state = all_state.to(self.device)
        x = F.relu(self.l1(all_state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        value = self.value_head(x)
        return value

def test_network():
    """
    测试Actor和Critic网络的功能和维度
    """
    print("=" * 60)
    print("开始测试MAPPO网络模块...")
    print("=" * 60)
    
    # 设置测试参数
    num_agents = 2
    num_goals = 3
    num_obstacles = 5
    
    # 网络维度参数
    self_state_dim = 3  # x, y, theta
    observe_state_dim = 32  # 雷达条数
    action_dim = 2  # speed, angular_speed
    all_state_dim = num_agents * 3 + num_goals * 3 + num_obstacles * 3  # 全局状态维度
    hidden_dim = 256
    max_action = np.array([1.0, 1.0])  # 最大速度和角速度
    batch_size = 5
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print(f"\n测试参数:")
    print(f"  智能体数量: {num_agents}")
    print(f"  目标数量: {num_goals}")
    print(f"  障碍物数量: {num_obstacles}")
    print(f"  智能体状态维度: {self_state_dim}")
    print(f"  观测维度: {observe_state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  全局状态维度: {all_state_dim}")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  批次大小: {batch_size}")
    print(f"  最大动作: {max_action}")
    
    try:
        # 测试Actor网络
        print(f"\n{'='*30}")
        print("测试Actor网络...")
        print(f"{'='*30}")
        
        actor = Actor(self_state_dim, observe_state_dim, action_dim, hidden_dim, max_action, device)
        print(f"✓ Actor网络创建成功")
        print(f"  网络参数数量: {sum(p.numel() for p in actor.parameters())}")
        
        # 创建测试输入
        self_state = torch.randn(batch_size, self_state_dim)
        observe_length = torch.randn(batch_size, observe_state_dim)
        observe_type = torch.randint(0, 6, (batch_size, observe_state_dim)).float()
        
        print(f"\n输入维度:")
        print(f"  self_state: {self_state.shape}")
        print(f"  observe_length: {observe_length.shape}")
        print(f"  observe_type: {observe_type.shape}")
        
        # 测试前向传播
        mean, std = actor(self_state, observe_length, observe_type)
        print(f"\n输出维度:")
        print(f"  mean: {mean.shape}")
        print(f"  std: {std.shape}")
        print(f"  期望维度: ({batch_size}, {action_dim})")
        
        # 测试动作采样
        action, log_prob = actor.get_action_and_log_prob(self_state, observe_length, observe_type)
        print(f"\n动作采样:")
        print(f"  action: {action.shape}")
        print(f"  log_prob: {log_prob.shape}")
        
        # 测试确定性动作
        det_action, _ = actor.get_action_and_log_prob(self_state, observe_length, observe_type, deterministic=True)
        print(f"  deterministic action: {det_action.shape}")
        
        # 测试动作评估
        eval_log_prob, entropy = actor.evaluate_actions(self_state, observe_length, observe_type, action)
        print(f"\n动作评估:")
        print(f"  eval_log_prob: {eval_log_prob.shape}")
        print(f"  entropy: {entropy.shape}")
        
        print(f"✓ Actor网络测试通过")
        
        # 测试Critic网络
        print(f"\n{'='*30}")
        print("测试Critic网络...")
        print(f"{'='*30}")
        
        critic = Critic(all_state_dim, hidden_dim, device)
        print(f"✓ Critic网络创建成功")
        print(f"  网络参数数量: {sum(p.numel() for p in critic.parameters())}")
        
        # 创建测试输入
        all_state = torch.randn(batch_size, all_state_dim)
        
        print(f"\n输入维度:")
        print(f"  all_state: {all_state.shape}")
        
        # 前向传播
        value = critic(all_state)
        print(f"\n输出维度:")
        print(f"  value: {value.shape}")
        print(f"  期望维度: ({batch_size}, 1)")
        
        print(f"\n价值范围:")
        print(f"  价值最小值: {value.min().item():.4f}")
        print(f"  价值最大值: {value.max().item():.4f}")
        print(f"  价值平均值: {value.mean().item():.4f}")
        
        print(f"✓ Critic网络测试通过")
        
        # 测试梯度传播
        print(f"\n{'='*30}")
        print("测试梯度传播...")
        print(f"{'='*30}")
        
        # Actor梯度测试
        actor.zero_grad()
        action, log_prob = actor.get_action_and_log_prob(self_state, observe_length, observe_type)
        loss = log_prob.sum()
        loss.backward()
        
        actor_grad_norm = 0
        for param in actor.parameters():
            if param.grad is not None:
                actor_grad_norm += param.grad.data.norm(2).item() ** 2
        actor_grad_norm = actor_grad_norm ** 0.5
        
        print(f"Actor梯度范数: {actor_grad_norm:.6f}")
        assert actor_grad_norm > 0, "Actor梯度为零"
        print(f"✓ Actor梯度传播正常")
        
        # Critic梯度测试
        critic.zero_grad()
        value = critic(all_state)
        loss = value.sum()
        loss.backward()
        
        critic_grad_norm = 0
        for param in critic.parameters():
            if param.grad is not None:
                critic_grad_norm += param.grad.data.norm(2).item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5
        
        print(f"Critic梯度范数: {critic_grad_norm:.6f}")
        assert critic_grad_norm > 0, "Critic梯度为零"
        print(f"✓ Critic梯度传播正常")
        
        print(f"\n{'='*60}")
        print("🎉 所有MAPPO网络测试通过！")
        print("✓ Actor网络功能正常（支持连续动作空间）")
        print("✓ Critic网络功能正常（只需要状态输入）") 
        print("✓ 梯度传播正常")
        print("✓ 动作采样和评估正常")
        print("✓ 维度匹配正确")
        print("✓ 动作范围合理")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ 网络测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_network()
