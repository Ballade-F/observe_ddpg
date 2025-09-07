import numpy as np
import torch
import json
import os
import logging
import time
from datetime import datetime

from maddpg import AgentTeam
from data_load import MultiAgentReplayBuffer
from sim_env import SimEnv

# 创建日志目录
if not os.path.exists('log'):
    os.makedirs('log')

# 创建模型保存目录
if not os.path.exists('models'):
    os.makedirs('models')

def setup_logging():
    """设置日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = f"log/train_{timestamp}"
    
    # 创建带时间戳的日志文件夹
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{log_dir}/training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 返回logger和日志目录路径
    return logging.getLogger(__name__), log_dir

def add_exploration_noise(actions, noise_std):
    """添加探索噪声"""
    noise = np.random.normal(0, noise_std, size=actions.shape)
    return actions + noise

def get_global_state(env):
    """构建全局状态"""
    # 全局状态包括：所有agent状态 + 所有目标状态 + 所有障碍物状态
    agent_states = env.agentState.flatten()  # (num_agents * 3,)
    goal_states = env.goalState.flatten()    # (num_goals * 3,)
    obstacle_states = env.obstacles.flatten() # (num_obstacles * 3,)
    
    global_state = np.concatenate([agent_states, goal_states, obstacle_states])
    return global_state

def train():
    # 加载配置
    config_path = 'F:\MARL\observe_ddpg\config.json'
    config = json.load(open(config_path, 'r'))
    
    # 设置日志
    logger, log_dir = setup_logging()
    
    # 在日志文件夹中保存配置文件副本
    config_copy_path = f"{log_dir}/config.json"
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("开始训练 MADDPG...")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"配置文件已保存到: {config_copy_path}")
    logger.info(f"配置参数: {config}")
    
    # 设置随机种子
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建环境
    env = SimEnv(config)
    logger.info(f"环境创建成功 - Agent数量: {env.kNumAgents}, 目标数量: {env.kNumGoals}, 障碍物数量: {env.kNumObstacles}")
    
    # 计算状态和动作维度
    agent_state_dim = 3  # x, y, theta
    observe_dim = env.kNumRadars  # 雷达观测维度
    action_dim = 2  # speed, angular_speed
    all_state_dim = env.kNumAgents * 3 + env.kNumGoals * 3 + env.kNumObstacles * 3
    max_action = np.array([env.kMaxSpeed, env.kMaxAngularSpeed])
    
    logger.info(f"状态维度 - Agent状态: {agent_state_dim}, 观测维度: {observe_dim}, 动作维度: {action_dim}")
    logger.info(f"全局状态维度: {all_state_dim}, 最大动作: {max_action}")
    
    # 创建智能体团队
    agent_team = AgentTeam(
        agent_state_dim=agent_state_dim,
        observe_dim=observe_dim,
        all_state_dim=all_state_dim,
        action_dim=action_dim,
        num_agents=env.kNumAgents,
        max_action=max_action,
        device=device,
        batch_size=config["training"]["batch_size"],
        gamma=config["training"]["gamma"],
        tau=config["training"]["tau"],
        lr=config["training"]["learning_rate"],
        lr_decay=config["training"]["lr_decay"],
        min_lr=config["training"]["min_learning_rate"]
    )
    
    logger.info("智能体团队创建成功")
    
    # 创建经验回放缓冲区
    replay_buffer = MultiAgentReplayBuffer(
        state_dim=all_state_dim,
        observe_dim=observe_dim,
        action_dim=action_dim,
        num_agents=env.kNumAgents,
        max_size=config["training"]["buffer_size"]
    )
    logger.info("经验回放缓冲区创建成功")
    
    # 训练参数
    max_episodes = config["training"]["max_episodes"]
    max_steps = config["training"]["max_steps_per_episode"]
    save_interval = config["training"]["save_interval"]
    log_interval = config["training"]["log_interval"]
    
    # 噪声参数
    noise_std = config["training"]["noise_std"]
    noise_decay = config["training"]["noise_decay"]
    min_noise_std = config["training"]["min_noise_std"]
    smart_noise_control = config["training"]["smart_noise_control"]
    noise_disabled = False  # 智能噪声控制标志
    
    # 记录训练过程的变量（使用固定长度列表节省内存）
    from collections import deque
    episode_rewards = deque(maxlen=log_interval)
    episode_steps = deque(maxlen=log_interval)
    success_episodes = deque(maxlen=log_interval)
    completed_goals = deque(maxlen=log_interval)  # 记录每个episode完成的任务点个数
    
    # 用于跟踪最佳成功率的变量
    best_success_rate = 0.0
    episodes_since_last_save = 0
    
    # 保存初始环境状态图
    logger.info("保存初始环境状态图...")
    try:
        # 创建图像保存目录
        if not os.path.exists('images'):
            os.makedirs('images')
        
        # 保存初始状态图
        initial_plot_path = "images/initial_environment_state.png"
        env.plot_environment(save_path=initial_plot_path, fig_size=(12, 10), dpi=150)
        logger.info(f"初始环境状态图已保存到: {initial_plot_path}")
    except Exception as e:
        logger.warning(f"保存初始环境状态图失败: {str(e)}")
    
    logger.info("开始训练循环...")
    start_time = time.time()
    
    for episode in range(max_episodes):
        episode_start_time = time.time()
        env.reset()
        
        episode_reward = 0
        episode_step = 0
        success = False
        
        # 获取初始状态
        current_global_state = get_global_state(env)
        current_observe_lengths = env.observeStateL
        current_observe_types = env.observeStateType
        
        for step in range(max_steps):
            # 选择动作
            actions = agent_team.choose_action(
                env.agentState,
                current_observe_lengths,
                current_observe_types
            )
            
            # 添加探索噪声（智能噪声控制）
            if not noise_disabled and noise_std > min_noise_std:
                actions = add_exploration_noise(actions, noise_std)
                # 限制动作范围
                actions[:, 0] = np.clip(actions[:, 0], -env.kMaxSpeed, env.kMaxSpeed)
                actions[:, 1] = np.clip(actions[:, 1], -env.kMaxAngularSpeed, env.kMaxAngularSpeed)
            
            # 执行动作
            next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions)
            
            # 智能噪声控制：检查是否有任务点被完成
            if smart_noise_control and not noise_disabled:
                # 检查是否有任何目标被完成（goalState[:, 2] > 0.5表示已完成）
                if np.any(env.goalState[:, 2] > 0.5):
                    noise_disabled = True
                    logger.info(f"Episode {episode+1}, Step {step+1}: 检测到任务点完成，停止添加噪声")
            
            # 获取下一步全局状态
            next_global_state = get_global_state(env)
            
            # 存储经验
            replay_buffer.store(
                state=current_global_state,
                observe_lengths=current_observe_lengths,
                observe_types=current_observe_types,
                actions=actions,
                rewards=reward,
                next_state=next_global_state,
                next_observe_lengths=next_observe_lengths,
                next_observe_types=next_observe_types,
                done=int(done)
            )
            
            # 更新状态
            current_global_state = next_global_state
            current_observe_lengths = next_observe_lengths
            current_observe_types = next_observe_types
            
            episode_reward += reward
            episode_step += 1
            
            # 学习
            if replay_buffer.size >= agent_team.batch_size:
                agent_team.learn(replay_buffer)
            
            if done:
                # 检查是否成功（所有目标都被到达）
                if np.all(env.goalState[:, 2] > 0.5):
                    success = True
                break
        
        # 统计完成的任务点个数
        num_completed_goals = np.sum(env.goalState[:, 2] > 0.5)

        
        # 衰减噪声（如果没有启用智能噪声控制或者噪声未被禁用）
        if not smart_noise_control or not noise_disabled:
            noise_std = max(min_noise_std, noise_std * noise_decay)
        
        # 记录训练数据（deque自动维护固定长度）
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        success_episodes.append(success)
        completed_goals.append(num_completed_goals)
        
        episode_time = time.time() - episode_start_time
        episodes_since_last_save += 1
        
        # 日志记录
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards)  # deque已经是固定长度
            avg_steps = np.mean(episode_steps)
            success_rate = np.mean(success_episodes)
            avg_completed_goals = np.mean(completed_goals)
            current_lr = agent_team.get_current_learning_rate()
            # 学习率衰减
            agent_team.decay_learning_rate()
            
            logger.info(f"Episode {episode+1}/{max_episodes} - "
                       f"平均奖励: {avg_reward:.3f}, "
                       f"平均步数: {avg_steps:.1f}, "
                       f"成功率: {success_rate:.3f}, "
                       f"平均完成任务点: {avg_completed_goals:.2f}, "
                       f"学习率: {current_lr:.6f}, "
                       f"噪声标准差: {noise_std:.4f}, "
                       f"用时: {episode_time:.2f}s")
            
            # 检查是否需要保存模型（成功率提升或达到保存间隔）
            should_save = False
            save_reason = ""
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                should_save = True
                save_reason = f"成功率提升至 {success_rate:.3f}"
                episodes_since_last_save = 0
            elif episodes_since_last_save >= save_interval:
                should_save = True
                save_reason = f"定期保存（{save_interval}轮间隔）"
                episodes_since_last_save = 0
            
            if should_save:
                model_dir = f"models/episode_{episode+1}_sr_{success_rate:.3f}"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                # 保存所有actor和critic网络
                for i, actor in enumerate(agent_team.actors):
                    torch.save(actor.state_dict(), f"{model_dir}/actor_{i}.pth")
                torch.save(agent_team.critic.state_dict(), f"{model_dir}/critic.pth")
                
                logger.info(f"模型已保存到 {model_dir} - {save_reason}")
    
    total_time = time.time() - start_time
    logger.info(f"训练完成！总用时: {total_time:.2f}s")
    
    # 保存最终模型
    final_model_dir = "models/final"
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    for i, actor in enumerate(agent_team.actors):
        torch.save(actor.state_dict(), f"{final_model_dir}/actor_{i}.pth")
    torch.save(agent_team.critic.state_dict(), f"{final_model_dir}/critic.pth")
    
    # 输出最终训练统计（基于最近的log_interval轮数据）
    final_avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
    final_avg_steps = np.mean(episode_steps) if len(episode_steps) > 0 else 0.0
    final_success_rate = np.mean(success_episodes) if len(success_episodes) > 0 else 0.0
    final_avg_completed_goals = np.mean(completed_goals) if len(completed_goals) > 0 else 0.0
    final_lr = agent_team.get_current_learning_rate()
    
    logger.info("=== 最终训练统计 ===")
    logger.info(f"最后{len(episode_rewards)}轮平均奖励: {final_avg_reward:.3f}")
    logger.info(f"最后{len(episode_steps)}轮平均步数: {final_avg_steps:.1f}")
    logger.info(f"最后{len(success_episodes)}轮成功率: {final_success_rate:.3f}")
    logger.info(f"最后{len(completed_goals)}轮平均完成任务点: {final_avg_completed_goals:.2f}")
    logger.info(f"最终学习率: {final_lr:.6f}")
    logger.info(f"最终噪声标准差: {noise_std:.4f}")
    logger.info(f"历史最佳成功率: {best_success_rate:.3f}")
    logger.info(f"总训练轮数: {max_episodes}")
    if len(episode_rewards) > 0:
        logger.info(f"最近轮次最高奖励: {np.max(episode_rewards):.3f}")
        logger.info(f"最近轮次最低奖励: {np.min(episode_rewards):.3f}")
    if len(completed_goals) > 0:
        logger.info(f"最近轮次最多完成任务点: {int(np.max(completed_goals))}")
        logger.info(f"最近轮次最少完成任务点: {int(np.min(completed_goals))}")
    logger.info("训练结束！")

if __name__ == "__main__":
    train()
