import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import warnings
warnings.filterwarnings('ignore')

from mappo_class import MAPPO
from sim_env import SimEnv, RadarObserveType

def get_global_state(env: SimEnv):
    """构建全局状态"""
    # 全局状态包括：所有agent状态 + 所有目标状态 + 所有障碍物状态
    agent_states = env.agentState.flatten()  # (num_agents * 3,)
    goal_states = env.goalState.flatten()    # (num_goals * 3,)
    obstacle_states = env.obstacles.flatten() # (num_obstacles * 3,)
    
    global_state = np.concatenate([agent_states, goal_states, obstacle_states])
    return global_state


def plot_realtime_environment(env: SimEnv, ax, frame_num, step_count, reward_history, goal_completion, critic_values=None, actions=None, action_logprobs=None):
    """
    实时绘制环境状态
    Args:
        env: 环境对象
        ax: matplotlib轴对象
        frame_num: 当前帧数
        step_count: 步数计数
        reward_history: 奖励历史
        goal_completion: 目标完成情况
        critic_values: Critic值历史
        actions: 当前动作
        action_logprobs: 动作对数概率
    """
    ax.clear()
    
    # 设置地图边界
    ax.set_xlim(env.kMinX - 0.5, env.kMaxX + 0.5)
    ax.set_ylim(env.kMinY - 0.5, env.kMaxY + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Multi-Agent Environment Evaluation - Step: {step_count}, Frame: {frame_num}')
    
    # 绘制地图边界
    boundary_rect = plt.Rectangle((env.kMinX, env.kMinY), 
                                env.kMaxX - env.kMinX, 
                                env.kMaxY - env.kMinY,
                                fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(boundary_rect)
    
    # 绘制障碍物
    for obs_x, obs_y, obs_r in env.obstacles:
        obstacle_circle = plt.Circle((obs_x, obs_y), obs_r, 
                                   color='gray', alpha=0.7)
        ax.add_patch(obstacle_circle)
    
    # 绘制目标点
    for i, (goal_x, goal_y, finish_flag) in enumerate(env.goalState):
        if finish_flag > 0.5:  # 已完成的目标
            goal_circle = plt.Circle((goal_x, goal_y), env.kGoalThreshold, 
                                   color='green', alpha=0.6)
        else:  # 未完成的目标
            goal_circle = plt.Circle((goal_x, goal_y), env.kGoalThreshold, 
                                   color='red', alpha=0.6)
        ax.add_patch(goal_circle)
        
        # 在目标中心添加标号
        ax.text(goal_x, goal_y, f'G{i}', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    # 绘制智能体
    colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
    for i, (agent_x, agent_y, agent_theta) in enumerate(env.agentState):
        color = colors[i % len(colors)]
        
        # 绘制智能体主体（圆形）
        if env.collision_flag[i]:
            # 碰撞的智能体用红色边框
            agent_circle = plt.Circle((agent_x, agent_y), env.kAgentRadius, 
                                    color=color, alpha=0.8, edgecolor='red', linewidth=3)
        else:
            agent_circle = plt.Circle((agent_x, agent_y), env.kAgentRadius, 
                                    color=color, alpha=0.8)
        ax.add_patch(agent_circle)
        
        # 绘制智能体朝向箭头
        arrow_length = env.kAgentRadius * 1.5
        dx = arrow_length * np.cos(agent_theta)
        dy = arrow_length * np.sin(agent_theta)
        ax.arrow(agent_x, agent_y, dx, dy, 
                head_width=env.kAgentRadius*0.3, head_length=env.kAgentRadius*0.2, 
                fc=color, ec=color, alpha=0.9)
        
        # 在智能体中心添加标号
        ax.text(agent_x, agent_y, f'A{i}', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # 在智能体下方显示该智能体的奖励、Critic值、动作和概率
        agent_info = f'A{i}'
        if len(reward_history) > 0:
            # 计算该智能体的累积奖励
            agent_cumulative_reward = np.sum([r[i] for r in reward_history if len(r) > i])
            agent_info += f'\nR: {agent_cumulative_reward:.2f}'
        
        if critic_values is not None and len(critic_values) > 0:
            # 显示该智能体的当前Critic值
            current_agent_critic = critic_values[-1][i] if len(critic_values[-1]) > i else 0.0
            agent_info += f'\nC: {current_agent_critic:.2f}'
        
        # 显示当前动作
        if actions is not None and len(actions) > i:
            speed = actions[i, 0]
            angular_speed = actions[i, 1]
            agent_info += f'\nv: {speed:.2f}, ω: {angular_speed:.2f}'
        
        # 显示动作概率
        if action_logprobs is not None and len(action_logprobs) > i:
            # 将对数概率转换为概率
            v_prob = np.exp(action_logprobs[i,0])
            w_prob = np.exp(action_logprobs[i,1])
            agent_info += f'\nPv: {v_prob:.3f}, Pw: {w_prob:.3f}'
        
        # 在智能体下方显示信息
        ax.text(agent_x, agent_y - env.kAgentRadius * 2.5, agent_info, 
               ha='center', va='top', fontsize=6.5, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black'),
               color='white', fontweight='bold')
    
    # 绘制雷达线
    radar_colors = {
        RadarObserveType.Empty.value: 'lightgray',
        RadarObserveType.Obstacle.value: 'red', 
        RadarObserveType.FinishedGoal.value: 'lightgreen',
        RadarObserveType.UnfinishedGoal.value: 'orange',
        RadarObserveType.Agent.value: 'lightblue'
    }
    
    for i, (agent_x, agent_y, agent_theta) in enumerate(env.agentState):
        for j in range(env.kNumRadars):
            # 计算雷达线角度
            radar_angle = env.radar_angle_range[j] + agent_theta
            
            # 获取雷达探测距离和类型
            radar_dist = env.observeStateL[i, j]
            radar_type = env.observeStateType[i, j]
            
            # 计算雷达线终点坐标
            end_x = agent_x + radar_dist * np.cos(radar_angle)
            end_y = agent_y + radar_dist * np.sin(radar_angle)
            
            # 获取对应类型的颜色
            line_color = radar_colors.get(radar_type, 'gray')
            
            # 绘制雷达线
            ax.plot([agent_x, end_x], [agent_y, end_y], 
                   color=line_color, alpha=0.4, linewidth=0.8)
    
    # 添加状态信息文本
    completed_goals = np.sum(env.goalState[:, 2] > 0.5)
    total_goals = len(env.goalState)
    info_text = f'Completed Goals: {completed_goals}/{total_goals}'
    
    # 计算总体统计信息
    if len(reward_history) > 0:
        # 计算所有智能体的总累积奖励
        total_reward = np.sum(reward_history)
        info_text += f'\nTotal Reward: {total_reward:.2f}'
    
    if critic_values is not None and len(critic_values) > 0:
        # 显示所有智能体的平均Critic值
        all_current_critics = critic_values[-1] if len(critic_values[-1]) > 0 else []
        avg_current_critic = np.mean(all_current_critics) if len(all_current_critics) > 0 else 0.0
        info_text += f'\nAvg Critic: {avg_current_critic:.3f}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def evaluate():
    """主评估函数"""
    # 加载配置
    config_path = './config.json'
    config = json.load(open(config_path, 'r'))

    # 加载模型
    model_dir = './models/train_2025-10-06-00-19'
    model_version = 2.0
    if not os.path.exists(model_dir):
        print(f"错误：找不到模型目录 {model_dir}")
        return
    actor_path = os.path.join(model_dir, f"actor_episode_{model_version}.pth")
    critic_path = os.path.join(model_dir, f"critic_episode_{model_version}.pth")
    
    print("开始评估 MAPPO...")
    
    # 设置随机种子以保证可重现性
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device(config["evaluation"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    # env0 = SimEnv(config)
    env = SimEnv(config)
    print(f"环境创建成功 - Agent数量: {env.kNumAgents}, 目标数量: {env.kNumGoals}, 障碍物数量: {env.kNumObstacles}")
    
    # 计算状态和动作维度
    agent_state_dim = 3  # x, y, theta
    observe_dim = env.kNumRadars  # 雷达观测维度
    action_dim = 2  # speed, angular_speed
    all_state_dim = env.kNumAgents * 3 + env.kNumGoals * 3 + env.kNumObstacles * 3
    max_action = np.array([env.kMaxSpeed, env.kMaxAngularSpeed])
    
    print(f"状态维度 - Agent状态: {agent_state_dim}, 观测维度: {observe_dim}, 动作维度: {action_dim}")
    print(f"全局状态维度: {all_state_dim}, 最大动作: {max_action}")
    
    # 参数设置
    actor_lr = config["training"]["actor_lr"]
    critic_lr = config["training"]["critic_lr"]
    total_episodes = config["training"]["total_episodes"]
    hidden_dim = config["network"]["hidden_dim"]
    gamma = config["training"]["gamma"]
    lamda = config["training"]["lamda"]
    eps = config["training"]["epsilon"]
    step_max = config["training"]["step_max"]

    # 创建智能体团队
    mappo = MAPPO(
        agent_n=env.kNumAgents,
        all_state_dim=all_state_dim,
        self_state_dim=agent_state_dim,
        observe_state_dim=observe_dim,
        action_dim=action_dim,
        max_action=max_action,
        gamma=gamma,
        lamda=lamda,
        epsilon=eps,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        device=device
    )

    try:
        mappo.load_model(actor_path, critic_path)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return
    
    # 评估参数
    animation_interval = 50
    
    # 初始化环境
    env.reset()
    print("环境初始化完成，开始评估...")
    
    # 用于记录评估过程的变量
    step_count = 0
    reward_history = []
    goal_completion_history = []
    critic_values_history = []
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update_animation(frame):
        nonlocal step_count, reward_history, goal_completion_history, critic_values_history
        
        if step_count >= step_max:
            ani.event_source.stop()
            print(f"达到最大步数 {step_max}，评估结束")
            return []
        
        # 选择动作（不添加噪声）
        actions, a_logprob = mappo.choose_action(
            env.agentState,
            env.observeStateL,
            env.observeStateType
        )
        
        # 计算Critic值
        global_state = get_global_state(env)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        actions_tensor = torch.FloatTensor(actions.flatten()).unsqueeze(0).to(device)
        
        with torch.no_grad():
            critic_value = mappo.critic(global_state_tensor).cpu().numpy().flatten() #critic_value: [agent_n]
        critic_values_history.append(critic_value)
        
        # 执行动作
        next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions)
        
        # 记录奖励和目标完成情况
        reward_history.append(reward)
        completed_goals = np.sum(env.goalState[:, 2] > 0.5)
        goal_completion_history.append(completed_goals)
        
        # 更新步数
        step_count += 1
        
        # 绘制当前状态（传递动作和概率信息）
        plot_realtime_environment(env, ax, frame, step_count, reward_history, goal_completion_history, 
                                 critic_values_history, actions, a_logprob)
        
        # 检查是否完成任务
        if done:
            ani.event_source.stop()
            # 计算所有智能体的总奖励
            total_reward = np.sum(reward_history)
            success = np.all(env.goalState[:, 2] > 0.5)
            
            # 计算每个智能体的累积奖励
            agent_rewards = [np.sum([r[i] for r in reward_history]) for i in range(len(reward_history[0]))]
            
            print(f"\n=== 评估结束 ===")
            print(f"总步数: {step_count}")
            print(f"总累积奖励: {total_reward:.2f}")
            for i, agent_reward in enumerate(agent_rewards):
                print(f"智能体 {i} 累积奖励: {agent_reward:.2f}")
            print(f"完成目标: {completed_goals}/{len(env.goalState)}")
            print(f"任务成功: {'是' if success else '否'}")
            
            # 在图上显示最终结果
            avg_critic = np.mean([np.mean(cv) for cv in critic_values_history]) if critic_values_history else 0.0
            
            # 构建每个智能体的奖励信息
            agent_reward_text = '\n'.join([f'Agent {i} Reward: {agent_rewards[i]:.2f}' for i in range(len(agent_rewards))])
            
            result_text = f'Evaluation Complete!\nTotal Steps: {step_count}\nTotal Reward: {total_reward:.2f}\n{agent_reward_text}\nCompleted Goals: {completed_goals}/{len(env.goalState)}\nAvg Critic Value: {avg_critic:.3f}\nTask Success: {"Yes" if success else "No"}'
            ax.text(0.5, 0.02, result_text, transform=ax.transAxes, 
                   ha='center', va='bottom', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        return []
    
    # 创建并运行动画
    ani = animation.FuncAnimation(fig, update_animation, frames=step_max, 
                                 interval=animation_interval, repeat=False)
    
    # # 保存动画（可选）
    # if config["evaluation"].get("save_animation", False):
    #     animation_path = config["evaluation"].get("animation_path", "evaluation_animation.gif")
    #     print(f"正在保存动画到: {animation_path}")
    #     ani.save(animation_path, writer='pillow', fps=20)
    #     print("动画保存完成")
    
    plt.show()


if __name__ == '__main__':
    evaluate()
