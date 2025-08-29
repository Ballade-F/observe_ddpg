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
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
        self.to(device)
    
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
        x = torch.mul(self.max_action.to(self.device), torch.tanh(self.l3(x)))
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
        self.l3 = nn.Linear(hidden_dim, 1)
        
        self.to(device)
    
    def forward(self, all_state: torch.Tensor, all_action: torch.Tensor) -> torch.Tensor:
        all_state = all_state.to(self.device)
        all_action = all_action.to(self.device)
        x = torch.cat([all_state, all_action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
def test_network():
    self_state_dim = 3
    observe_state_dim = 32
    action_dim = 2
    all_action_dim = 4
    all_state_dim = 10
    hidden_dim = 256
    max_action = np.array([1.0, 1.0])
    batch_size = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    actor = Actor(self_state_dim, observe_state_dim, action_dim, hidden_dim, max_action, device)
    critic = Critic(all_state_dim , all_action_dim, hidden_dim, device)

    self_state = torch.randn(batch_size, self_state_dim)
    observe_length = torch.randn(batch_size, observe_state_dim)
    observe_type = torch.randn(batch_size, observe_state_dim)
    action = actor(self_state, observe_length, observe_type)
    q = critic(self_state, action)
    print(q)
    print(action)

if __name__ == "__main__":
    test_network()