import numpy as np
import torch
from maddpg import MADDPG
from data_load import MultiAgentReplayBuffer
from sim_env import SimEnv
import json

def main():
    # 环境配置
    config = {
        "map": {
            "kMaxX": 5.0,
            "kMaxY": 5.0,
            "kMinX": -5.0,
            "kMinY": -5.0,
            "kMaxObsR": 1.5,
            "kMinObsR": 0.5,
            "kNumAgents": 2,  # 2个智能体
            "kNumGoals": 3,
            "kNumObstacles": 2,
            "kTimeStep": 0.1
        },
        "agent": {
            "kAgentRadius": 0.2,
            "kMaxSpeed": 1.0,
            "kMaxAngularSpeed": 1.0,
            "kNumRadars": 32,
            "kMaxRadarDist": 3.0
        },
        "goal": {
            "kGoalThreshold": 0.5
        },
        "reward": {
            "kTimeReward": -0.01,
            "kCollisionReward": -1.0,
            "kGoalReward": 1.0,
            "kDistReward": 0.01
        }
    }
    
    # 环境和智能体参数
    num_agents = config["map"]["kNumAgents"]
    num_goals = config["map"]["kNumGoals"]
    num_obstacles = config["map"]["kNumObstacles"]
    state_dim = 3  # x, y, theta
    observe_dim = config["agent"]["kNumRadars"]
    action_dim = 2  # speed, angular_speed
    
    # 计算全局状态维度：所有agent的状态 + 目标状态 + 障碍物状态
    all_state_dim = num_agents * state_dim + num_goals * 2 + num_obstacles * 3
    
    max_action = np.array([config["agent"]["kMaxSpeed"], config["agent"]["kMaxAngularSpeed"]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化环境
    env = SimEnv(config)
    
    # 初始化MADDPG算法
    maddpg = MADDPG(
        state_dim=state_dim,
        observe_dim=observe_dim,
        action_dim=action_dim,
        all_state_dim=all_state_dim,
        num_agents=num_agents,
        max_action=max_action,
        device=device
    )
    
    # 初始化多智能体回放缓冲区
    replay_buffer = MultiAgentReplayBuffer(
        state_dim=state_dim,
        observe_dim=observe_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        max_size=100000
    )
    
    # 训练参数
    max_episodes = 1000
    max_steps = 200
    
    print(f"开始训练MADDPG算法，共{max_episodes}轮...")
    print(f"智能体数量: {num_agents}")
    print(f"状态维度: {state_dim}, 观测维度: {observe_dim}, 动作维度: {action_dim}")
    print(f"全局状态维度: {all_state_dim}")
    print(f"设备: {device}")
    
    for episode in range(max_episodes):
        # 重置环境
        env.reset()
        # 构建全局状态
        # 注意：当前sim_env只支持单个agent，这里需要扩展为多agent
        # 暂时用单个agent的状态来模拟多agent
        agent_states = np.tile(env.agentState, (num_agents, 1))  # (num_agents, 3)
        state = maddpg.build_global_state(agent_states, env.goalStartState.reshape(-1, 2), env.obstacles)
        observe_lengths = np.tile(env.observeStateL, (num_agents, 1))  # (num_agents, observe_dim)
        observe_types = np.tile(env.observeStateType, (num_agents, 1))  # (num_agents, observe_dim)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作 - 每个agent基于自己的状态选择动作
            current_agent_states = np.tile(env.agentState, (num_agents, 1))  # (num_agents, 3)
            actions = maddpg.choose_action(current_agent_states, observe_lengths, observe_types)
            
            # 执行动作 - 暂时只使用第一个agent的动作
            next_agent_state, next_goal_state, next_obstacles, next_observe_length, next_observe_type, rewards, done = env.step(actions[0:1])
            
            # 构建下一个全局状态
            next_agent_states = np.tile(next_agent_state, (num_agents, 1))  # (num_agents, 3)
            next_state = maddpg.build_global_state(next_agent_states, next_goal_state.reshape(-1, 2), next_obstacles)
            next_observe_lengths = np.tile(next_observe_length, (num_agents, 1))  # (num_agents, observe_dim)
            next_observe_types = np.tile(next_observe_type, (num_agents, 1))  # (num_agents, observe_dim)
            
            # 存储经验
            # 将单个agent的reward扩展为多个agent的reward
            multi_agent_rewards = np.tile(rewards, num_agents)  # (num_agents,)
            replay_buffer.store(
                state=state,
                observe_lengths=observe_lengths,
                observe_types=observe_types,
                actions=actions,
                rewards=multi_agent_rewards,
                next_state=next_state,
                next_observe_lengths=next_observe_lengths,
                next_observe_types=next_observe_types,
                done=done
            )
            
            # 更新状态
            state = next_state
            observe_lengths = next_observe_lengths
            observe_types = next_observe_types
            episode_reward += multi_agent_rewards.sum()  # 所有agent奖励之和
            
            # 学习
            if replay_buffer.size > maddpg.batch_size:
                maddpg.learn(replay_buffer)
            
            if done:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Buffer Size: {replay_buffer.size}")
    
    print("训练完成！")

if __name__ == "__main__":
    main() 