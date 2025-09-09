import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import warnings
warnings.filterwarnings('ignore')

from maddpg import AgentTeam
from sim_env import SimEnv


def plot_realtime_environment(env, ax, frame_num, step_count, reward_history, goal_completion):
    """
    实时绘制环境状态
    Args:
        env: 环境对象
        ax: matplotlib轴对象
        frame_num: 当前帧数
        step_count: 步数计数
        reward_history: 奖励历史
        goal_completion: 目标完成情况
    """
    ax.clear()
    
    # 设置地图边界
    ax.set_xlim(env.kMinX - 0.5, env.kMaxX + 0.5)
    ax.set_ylim(env.kMinY - 0.5, env.kMaxY + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(f'多智能体环境评估 - 步数: {step_count}, 帧数: {frame_num}')
    
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
    
    # 添加状态信息文本
    completed_goals = np.sum(env.goalState[:, 2] > 0.5)
    total_goals = len(env.goalState)
    info_text = f'已完成目标: {completed_goals}/{total_goals}'
    if len(reward_history) > 0:
        info_text += f'\n累积奖励: {sum(reward_history):.2f}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def evaluate():
    """主评估函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MADDPG多智能体评估')
    parser.add_argument('--config', type=str, default='evaluate_config.json', 
                       help='评估配置文件路径')
    parser.add_argument('--model_dir', type=str, default=None, 
                       help='模型文件夹路径（覆盖配置文件中的设置）')
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"错误：找不到配置文件 {args.config}")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("开始评估 MADDPG...")
    print(f"配置文件: {args.config}")
    print(f"配置参数: {config}")
    
    # 设置随机种子以保证可重现性
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device(config["evaluation"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
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
    
    # 创建智能体团队
    agent_team = AgentTeam(
        agent_state_dim=agent_state_dim,
        observe_dim=observe_dim,
        all_state_dim=all_state_dim,
        action_dim=action_dim,
        num_agents=env.kNumAgents,
        max_action=max_action,
        device=device
    )
    
    # 加载模型
    model_dir = args.model_dir if args.model_dir else config["evaluation"]["model_dir"]
    if not os.path.exists(model_dir):
        print(f"错误：找不到模型目录 {model_dir}")
        return
    
    try:
        agent_team.load_models(model_dir)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return
    
    # 评估参数
    max_steps = config["evaluation"]["max_steps"]
    animation_interval = config["evaluation"]["animation_interval"]
    
    # 初始化环境
    env.reset()
    print("环境初始化完成，开始评估...")
    
    # 用于记录评估过程的变量
    step_count = 0
    reward_history = []
    goal_completion_history = []
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update_animation(frame):
        nonlocal step_count, reward_history, goal_completion_history
        
        if step_count >= max_steps:
            ani.event_source.stop()
            print(f"达到最大步数 {max_steps}，评估结束")
            return []
        
        # 选择动作（不添加噪声）
        actions = agent_team.choose_action(
            env.agentState,
            env.observeStateL,
            env.observeStateType
        )
        
        # 执行动作
        next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions)
        
        # 记录奖励和目标完成情况
        reward_history.append(reward)
        completed_goals = np.sum(env.goalState[:, 2] > 0.5)
        goal_completion_history.append(completed_goals)
        
        # 更新步数
        step_count += 1
        
        # 绘制当前状态
        plot_realtime_environment(env, ax, frame, step_count, reward_history, goal_completion_history)
        
        # 检查是否完成任务
        if done:
            ani.event_source.stop()
            total_reward = sum(reward_history)
            success = np.all(env.goalState[:, 2] > 0.5)
            print(f"\n=== 评估结束 ===")
            print(f"总步数: {step_count}")
            print(f"累积奖励: {total_reward:.2f}")
            print(f"完成目标: {completed_goals}/{len(env.goalState)}")
            print(f"任务成功: {'是' if success else '否'}")
            
            # 在图上显示最终结果
            result_text = f'评估完成!\n总步数: {step_count}\n累积奖励: {total_reward:.2f}\n完成目标: {completed_goals}/{len(env.goalState)}\n任务成功: {"是" if success else "否"}'
            ax.text(0.5, 0.02, result_text, transform=ax.transAxes, 
                   ha='center', va='bottom', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=12, fontweight='bold')
        
        return []
    
    # 创建并运行动画
    ani = animation.FuncAnimation(fig, update_animation, frames=max_steps, 
                                 interval=animation_interval, repeat=False)
    
    # 保存动画（可选）
    if config["evaluation"].get("save_animation", False):
        animation_path = config["evaluation"].get("animation_path", "evaluation_animation.gif")
        print(f"正在保存动画到: {animation_path}")
        ani.save(animation_path, writer='pillow', fps=20)
        print("动画保存完成")
    
    plt.show()


if __name__ == '__main__':
    evaluate()
