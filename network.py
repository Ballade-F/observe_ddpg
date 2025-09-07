import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#TODO:以后需要考虑归一化的问题
#TODO:改成attention网络，尤其是critic。最重要考虑的是让网络与智能体数量、（雷达条数）无关
#TODO: 这并不是一个马尔科夫过程，智能体以往的观测应当影响当前的决策，需要考虑历史观测，可以试试SLAM中的关键帧思想

class Actor(nn.Module):
    def __init__(self, self_state_dim: int, observe_state_dim: int, action_dim: int, 
                 hidden_dim: int, max_action: np.ndarray, device: torch.device):
        super(Actor, self).__init__()
        self.max_action = torch.FloatTensor(max_action)
        self.device = device

        self.l1 = nn.Linear(self_state_dim + 2*observe_state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, action_dim)
        
        # Kaiming初始化
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Kaiming初始化"""
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        # 输出层使用较小的初始化
        nn.init.uniform_(self.l4.weight, -3e-3, 3e-3)
        nn.init.constant_(self.l4.bias, 0)
    
    def forward(self, self_state: torch.Tensor, observe_length: torch.Tensor, observe_type: torch.Tensor) -> torch.Tensor:
        '''
        self_state: (batch_size, self_state_dim) 智能体自身状态 x,y,theta
        observe_length: (batch_size, observe_state_dim) 雷达探测到的障碍物距离
        observe_type: (batch_size, observe_state_dim) 雷达探测到的障碍物类型
        '''
        self_state = self_state.to(self.device)
        observe_length = observe_length.to(self.device)
        observe_type = observe_type.to(self.device)
        x = torch.cat([self_state, observe_length, observe_type], dim=1)    
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = torch.mul(self.max_action.to(self.device), torch.tanh(self.l4(x)))
        return x

class Critic(nn.Module):
    def __init__(self, all_state_dim: int, all_action_dim: int, hidden_dim: int, device: torch.device):
        super(Critic, self).__init__()
        self.device = device

        # all_state_dim: 全局状态维度，包含所有agent状态、目标状态、障碍物状态
        # all_state_dim = agent_state_dim*num_agents + goal_state_dim*num_goals + obstacles_state_dim*num_obstacles
        # 即 3*num_agents + 3*num_goals + 3*num_obstacles
        self.l1 = nn.Linear(all_state_dim + all_action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
        
        # Kaiming初始化
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Kaiming初始化"""
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        # 输出层使用较小的初始化
        nn.init.uniform_(self.l4.weight, -3e-3, 3e-3)
        nn.init.constant_(self.l4.bias, 0)
    
    def forward(self, all_state: torch.Tensor, all_action: torch.Tensor) -> torch.Tensor:
        all_state = all_state.to(self.device)
        all_action = all_action.to(self.device)
        x = torch.cat([all_state, all_action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    
def test_network():
    """
    测试Actor和Critic网络的功能和维度
    """
    print("=" * 60)
    print("开始测试网络模块...")
    print("=" * 60)
    
    # 设置测试参数 - 基于config.json的默认配置
    num_agents = 2
    num_goals = 3
    num_obstacles = 5
    
    # 网络维度参数
    self_state_dim = 3  # x, y, theta
    observe_state_dim = 32  # 雷达条数
    action_dim = 2  # speed, angular_speed
    all_state_dim = num_agents * 3 + num_goals * 3 + num_obstacles * 3  # 全局状态维度
    all_action_dim = action_dim * num_agents  # 所有智能体的动作维度
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
    print(f"  全局动作维度: {all_action_dim}")
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
        observe_type = torch.randn(batch_size, observe_state_dim)
        
        print(f"\n输入维度:")
        print(f"  self_state: {self_state.shape}")
        print(f"  observe_length: {observe_length.shape}")
        print(f"  observe_type: {observe_type.shape}")
        
        # 前向传播
        action = actor(self_state, observe_length, observe_type)
        print(f"\n输出维度:")
        print(f"  action: {action.shape}")
        print(f"  期望维度: ({batch_size}, {action_dim})")
        
        
        # 检查维度是否正确
        assert action.shape == (batch_size, action_dim), f"Actor输出维度错误: 期望{(batch_size, action_dim)}, 实际{action.shape}"
        
        # 检查动作是否在合理范围内
        assert torch.all(action >= -torch.FloatTensor(max_action).to(device) - 1e-6), "动作值低于最小值"
        assert torch.all(action <= torch.FloatTensor(max_action).to(device) + 1e-6), "动作值高于最大值"
        
        print(f"✓ Actor网络测试通过")
        
        # 测试Critic网络
        print(f"\n{'='*30}")
        print("测试Critic网络...")
        print(f"{'='*30}")
        
        critic = Critic(all_state_dim, all_action_dim, hidden_dim, device)
        print(f"✓ Critic网络创建成功")
        print(f"  网络参数数量: {sum(p.numel() for p in critic.parameters())}")
        
        # 创建测试输入
        all_state = torch.randn(batch_size, all_state_dim)
        all_action = torch.randn(batch_size, all_action_dim)
        
        print(f"\n输入维度:")
        print(f"  all_state: {all_state.shape}")
        print(f"  all_action: {all_action.shape}")
        
        # 前向传播
        q_value = critic(all_state, all_action)
        print(f"\n输出维度:")
        print(f"  q_value: {q_value.shape}")
        print(f"  期望维度: ({batch_size}, 1)")
        
        print(f"\nQ值范围:")
        print(f"  Q值最小值: {q_value.min().item():.4f}")
        print(f"  Q值最大值: {q_value.max().item():.4f}")
        print(f"  Q值平均值: {q_value.mean().item():.4f}")
        
        # 检查维度是否正确
        assert q_value.shape == (batch_size, 1), f"Critic输出维度错误: 期望{(batch_size, 1)}, 实际{q_value.shape}"
        
        print(f"✓ Critic网络测试通过")
        
        # 测试梯度传播
        print(f"\n{'='*30}")
        print("测试梯度传播...")
        print(f"{'='*30}")
        
        # Actor梯度测试
        actor.zero_grad()
        action = actor(self_state, observe_length, observe_type)
        loss = action.sum()
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
        q_value = critic(all_state, all_action)
        loss = q_value.sum()
        loss.backward()
        
        critic_grad_norm = 0
        for param in critic.parameters():
            if param.grad is not None:
                critic_grad_norm += param.grad.data.norm(2).item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5
        
        print(f"Critic梯度范数: {critic_grad_norm:.6f}")
        assert critic_grad_norm > 0, "Critic梯度为零"
        print(f"✓ Critic梯度传播正常")
        
        # 测试多智能体场景
        print(f"\n{'='*30}")
        print("测试多智能体场景...")
        print(f"{'='*30}")
        
        # 创建多个Actor网络
        actors = []
        for i in range(num_agents):
            actors.append(Actor(self_state_dim, observe_state_dim, action_dim, hidden_dim, max_action, device))
        
        print(f"✓ 创建了{num_agents}个Actor网络")
        
        # 测试每个智能体的动作选择
        all_actions = []
        for i in range(num_agents):
            agent_state = torch.randn(batch_size, self_state_dim)
            agent_observe_length = torch.randn(batch_size, observe_state_dim)
            agent_observe_type = torch.randn(batch_size, observe_state_dim)
            
            agent_action = actors[i](agent_state, agent_observe_length, agent_observe_type)
            all_actions.append(agent_action)
            
            print(f"  智能体{i}动作维度: {agent_action.shape}")
        
        # 拼接所有动作
        combined_actions = torch.cat(all_actions, dim=1)
        print(f"合并后动作维度: {combined_actions.shape}")
        print(f"期望维度: ({batch_size}, {all_action_dim})")
        
        assert combined_actions.shape == (batch_size, all_action_dim), "多智能体动作拼接维度错误"
        print(f"✓ 多智能体动作拼接测试通过")
        
        # 测试与Critic的兼容性
        q_value_multi = critic(all_state, combined_actions)
        print(f"多智能体Critic输出维度: {q_value_multi.shape}")
        assert q_value_multi.shape == (batch_size, 1), "多智能体Critic输出维度错误"
        print(f"✓ 多智能体与Critic兼容性测试通过")
        
        print(f"\n{'='*60}")
        print("🎉 所有网络测试通过！")
        print("✓ Actor网络功能正常")
        print("✓ Critic网络功能正常") 
        print("✓ 梯度传播正常")
        print("✓ 多智能体场景兼容")
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