import numpy as np
import torch
import json
import os
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt

from mappo import MAPPOAgent
from sim_env import SimEnv

def setup_evaluation_logging():
    """设置评估日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = f"log/mappo_eval_{timestamp}"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{log_dir}/evaluation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_dir

def get_global_state(env):
    """构建全局状态"""
    agent_states = env.agentState.flatten()
    goal_states = env.goalState.flatten()
    obstacle_states = env.obstacles.flatten()
    
    global_state = np.concatenate([agent_states, goal_states, obstacle_states])
    return global_state

def evaluate_single_episode(agent: MAPPOAgent, config: dict, max_steps: int, 
                           render: bool = False, save_path: str = None) -> dict:
    """
    评估单个episode
    Args:
        agent: MAPPO智能体
        config: 环境配置
        max_steps: 最大步数
        render: 是否可视化
        save_path: 保存路径（如果需要保存轨迹图）
    Returns:
        episode统计信息
    """
    env = SimEnv(config)
    env.reset()
    
    episode_reward = 0
    episode_step = 0
    success = False
    
    # 记录轨迹（用于可视化）
    if render or save_path:
        trajectory = {
            'agent_positions': [],
            'goal_states': [],
            'rewards': []
        }
    
    # 获取初始状态
    current_global_state = get_global_state(env)
    current_agent_states = env.agentState
    current_observe_lengths = env.observeStateL
    current_observe_types = env.observeStateType
    
    for step in range(max_steps):
        # 记录轨迹
        if render or save_path:
            trajectory['agent_positions'].append(env.agentState.copy())
            trajectory['goal_states'].append(env.goalState.copy())
        
        # 选择动作（确定性策略）
        actions, _ = agent.choose_action(
            current_agent_states,
            current_observe_lengths,
            current_observe_types,
            deterministic=True
        )
        
        # 执行动作
        next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions)
        
        # 记录奖励
        if render or save_path:
            trajectory['rewards'].append(reward)
        
        # 更新状态
        current_global_state = get_global_state(env)
        current_agent_states = next_agent_state
        current_observe_lengths = next_observe_lengths
        current_observe_types = next_observe_types
        
        episode_reward += reward
        episode_step += 1
        
        if done:
            # 检查是否成功
            if np.all(env.goalState[:, 2] > 0.5):
                success = True
            break
    
    # 统计完成的任务点个数
    num_completed_goals = np.sum(env.goalState[:, 2] > 0.5)
    
    # 保存轨迹图
    if save_path and (render or save_path):
        plot_trajectory(env, trajectory, save_path)
    
    return {
        'reward': episode_reward,
        'steps': episode_step,
        'success': success,
        'completed_goals': num_completed_goals,
        'trajectory': trajectory if (render or save_path) else None
    }

def plot_trajectory(env, trajectory, save_path: str):
    """
    绘制智能体轨迹图
    Args:
        env: 环境实例
        trajectory: 轨迹数据
        save_path: 保存路径
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 绘制环境边界
        ax.set_xlim(env.kMinX - 0.5, env.kMaxX + 0.5)
        ax.set_ylim(env.kMinY - 0.5, env.kMaxY + 0.5)
        
        # 绘制障碍物
        for obs in env.obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.7, label='障碍物' if obs is env.obstacles[0] else "")
            ax.add_patch(circle)
        
        # 绘制目标点
        final_goals = trajectory['goal_states'][-1]
        for i, goal in enumerate(final_goals):
            color = 'green' if goal[2] > 0.5 else 'red'
            marker = 'o' if goal[2] > 0.5 else 'x'
            ax.scatter(goal[0], goal[1], c=color, s=100, marker=marker, 
                      label='已完成目标' if i == 0 and goal[2] > 0.5 else ('未完成目标' if i == 0 and goal[2] <= 0.5 else ""))
        
        # 绘制智能体轨迹
        colors = ['blue', 'orange', 'purple', 'brown', 'pink']
        for agent_id in range(env.kNumAgents):
            positions = np.array([pos[agent_id] for pos in trajectory['agent_positions']])
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=colors[agent_id % len(colors)], linewidth=2, alpha=0.8,
                   label=f'智能体{agent_id+1}轨迹')
            
            # 标记起始和结束位置
            ax.scatter(positions[0, 0], positions[0, 1], 
                      color=colors[agent_id % len(colors)], s=150, marker='s', alpha=0.8)
            ax.scatter(positions[-1, 0], positions[-1, 1], 
                      color=colors[agent_id % len(colors)], s=150, marker='^', alpha=0.8)
        
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title('MAPPO智能体轨迹图')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"绘制轨迹图失败: {str(e)}")

def evaluate():
    """主评估函数"""
    # 加载评估配置
    eval_config_path = './evaluate_config.json'
    if os.path.exists(eval_config_path):
        eval_config = json.load(open(eval_config_path, 'r'))
    else:
        # 使用默认配置
        eval_config = {
            "model_path": "./log/mappo_train_latest/best_model",
            "num_episodes": 100,
            "render": False,
            "save_trajectories": True,
            "max_trajectory_saves": 10
        }
        # 保存默认配置
        with open(eval_config_path, 'w', encoding='utf-8') as f:
            json.dump(eval_config, f, indent=2, ensure_ascii=False)
        print(f"已创建默认评估配置文件: {eval_config_path}")
    
    # 加载环境配置
    config_path = './config.json'
    config = json.load(open(config_path, 'r'))
    
    # 设置日志
    logger, log_dir = setup_evaluation_logging()
    
    logger.info("开始评估 MAPPO...")
    logger.info(f"评估日志目录: {log_dir}")
    logger.info(f"模型路径: {eval_config['model_path']}")
    logger.info(f"评估轮数: {eval_config['num_episodes']}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建环境（用于获取维度信息）
    env = SimEnv(config)
    
    # 计算状态和动作维度
    agent_state_dim = 3
    observe_dim = env.kNumRadars
    action_dim = 2
    all_state_dim = env.kNumAgents * 3 + env.kNumGoals * 3 + env.kNumObstacles * 3
    max_action = np.array([env.kMaxSpeed, env.kMaxAngularSpeed])
    
    # 创建MAPPO智能体
    agent = MAPPOAgent(
        agent_state_dim=agent_state_dim,
        observe_dim=observe_dim,
        all_state_dim=all_state_dim,
        action_dim=action_dim,
        num_agents=env.kNumAgents,
        max_action=max_action,
        device=device
    )
    
    # 加载训练好的模型
    try:
        agent.load_models(eval_config['model_path'])
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return
    
    # 评估参数
    num_episodes = eval_config['num_episodes']
    max_steps = config["training"]["max_steps_per_episode"]
    render = eval_config.get('render', False)
    save_trajectories = eval_config.get('save_trajectories', False)
    max_trajectory_saves = eval_config.get('max_trajectory_saves', 10)
    
    # 创建轨迹保存目录
    if save_trajectories:
        trajectory_dir = f"{log_dir}/trajectories"
        if not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)
    
    # 评估统计
    all_rewards = []
    all_steps = []
    success_count = 0
    all_completed_goals = []
    
    logger.info("开始评估循环...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        
        # 决定是否保存轨迹
        save_path = None
        if save_trajectories and episode < max_trajectory_saves:
            save_path = f"{trajectory_dir}/episode_{episode+1}.png"
        
        # 运行单个episode
        result = evaluate_single_episode(
            agent, config, max_steps, render, save_path
        )
        
        # 记录统计信息
        all_rewards.append(result['reward'])
        all_steps.append(result['steps'])
        all_completed_goals.append(result['completed_goals'])
        if result['success']:
            success_count += 1
        
        episode_time = time.time() - episode_start_time
        
        # 每10轮输出一次进度
        if (episode + 1) % 10 == 0:
            current_success_rate = success_count / (episode + 1)
            current_avg_reward = np.mean(all_rewards)
            current_avg_completed = np.mean(all_completed_goals)
            
            logger.info(f"Episode {episode+1}/{num_episodes} - "
                       f"成功率: {current_success_rate:.3f}, "
                       f"平均奖励: {current_avg_reward:.3f}, "
                       f"平均完成任务点: {current_avg_completed:.2f}, "
                       f"用时: {episode_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # 计算最终统计
    final_success_rate = success_count / num_episodes
    final_avg_reward = np.mean(all_rewards)
    final_std_reward = np.std(all_rewards)
    final_avg_steps = np.mean(all_steps)
    final_avg_completed_goals = np.mean(all_completed_goals)
    final_std_completed_goals = np.std(all_completed_goals)
    
    # 输出最终结果
    logger.info("=" * 60)
    logger.info("🎉 评估完成！")
    logger.info("=" * 60)
    logger.info(f"总评估轮数: {num_episodes}")
    logger.info(f"总用时: {total_time:.2f}s")
    logger.info(f"平均每轮用时: {total_time/num_episodes:.2f}s")
    logger.info("")
    logger.info("=== 性能统计 ===")
    logger.info(f"成功率: {final_success_rate:.3f} ({success_count}/{num_episodes})")
    logger.info(f"平均奖励: {final_avg_reward:.3f} ± {final_std_reward:.3f}")
    logger.info(f"平均步数: {final_avg_steps:.1f}")
    logger.info(f"平均完成任务点: {final_avg_completed_goals:.2f} ± {final_std_completed_goals:.2f}")
    logger.info(f"最高奖励: {np.max(all_rewards):.3f}")
    logger.info(f"最低奖励: {np.min(all_rewards):.3f}")
    logger.info(f"最多完成任务点: {int(np.max(all_completed_goals))}")
    logger.info(f"最少完成任务点: {int(np.min(all_completed_goals))}")
    
    # 保存评估结果
    results = {
        'success_rate': final_success_rate,
        'avg_reward': final_avg_reward,
        'std_reward': final_std_reward,
        'avg_steps': final_avg_steps,
        'avg_completed_goals': final_avg_completed_goals,
        'std_completed_goals': final_std_completed_goals,
        'max_reward': float(np.max(all_rewards)),
        'min_reward': float(np.min(all_rewards)),
        'max_completed_goals': int(np.max(all_completed_goals)),
        'min_completed_goals': int(np.min(all_completed_goals)),
        'total_episodes': num_episodes,
        'total_time': total_time,
        'model_path': eval_config['model_path']
    }
    
    results_path = f"{log_dir}/evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评估结果已保存到: {results_path}")
    
    if save_trajectories:
        logger.info(f"前{max_trajectory_saves}轮轨迹图已保存到: {trajectory_dir}")
    
    logger.info("评估结束！")

if __name__ == "__main__":
    evaluate()
