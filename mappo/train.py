import numpy as np
import torch
import json
import os
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from mappo import MAPPOAgent
from data_load import MultiEnvMAPPOBuffer
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
    log_dir = f"log/mappo_train_{timestamp}"
    
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

def get_global_state(env):
    """构建全局状态"""
    # 全局状态包括：所有agent状态 + 所有目标状态 + 所有障碍物状态
    agent_states = env.agentState.flatten()  # (num_agents * 3,)
    goal_states = env.goalState.flatten()    # (num_goals * 3,)
    obstacle_states = env.obstacles.flatten() # (num_obstacles * 3,)
    
    global_state = np.concatenate([agent_states, goal_states, obstacle_states])
    return global_state

def run_single_environment(env_id: int, config: dict, agent: MAPPOAgent, 
                          multi_buffer: MultiEnvMAPPOBuffer, max_steps: int) -> dict:
    """
    运行单个环境的episode
    Args:
        env_id: 环境ID
        config: 配置参数
        agent: MAPPO智能体
        multi_buffer: 多环境缓冲区
        max_steps: 最大步数
    Returns:
        episode统计信息
    """
    # 创建环境实例
    env = SimEnv(config)
    env.reset()
    
    episode_reward = 0
    episode_step = 0
    success = False
    
    # 获取初始状态
    current_global_state = get_global_state(env)
    current_agent_states = env.agentState
    current_observe_lengths = env.observeStateL
    current_observe_types = env.observeStateType
    
    for step in range(max_steps):
        # 获取当前状态的价值估计
        current_value = agent.get_value(current_global_state)
        
        # 选择动作
        actions, log_probs = agent.choose_action(
            current_agent_states,
            current_observe_lengths,
            current_observe_types,
            deterministic=False
        )
        
        # 执行动作
        next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions)
        
        # 获取下一步全局状态
        next_global_state = get_global_state(env)
        
        # 存储经验到对应环境的缓冲区
        multi_buffer.store(
            env_id=env_id,
            state=current_global_state,
            agent_states=current_agent_states,
            observe_lengths=current_observe_lengths,
            observe_types=current_observe_types,
            actions=actions,
            log_probs=log_probs,
            reward=reward,
            done=done,
            value=current_value
        )
        
        # 更新状态
        current_global_state = next_global_state
        current_agent_states = next_agent_state
        current_observe_lengths = next_observe_lengths
        current_observe_types = next_observe_types
        
        episode_reward += reward
        episode_step += 1
        
        if done:
            # 检查是否成功（所有目标都被到达）
            if np.all(env.goalState[:, 2] > 0.5):
                success = True
            break
    
    # 存储最终状态的价值估计
    final_value = agent.get_value(current_global_state) if not done else 0.0
    multi_buffer.store_final_value(env_id, final_value)
    
    # 统计完成的任务点个数
    num_completed_goals = np.sum(env.goalState[:, 2] > 0.5)
    
    return {
        'env_id': env_id,
        'reward': episode_reward,
        'steps': episode_step,
        'success': success,
        'completed_goals': num_completed_goals
    }

def train():
    # 加载配置
    config_path = './config.json'
    config = json.load(open(config_path, 'r'))
    
    # 设置日志
    logger, log_dir = setup_logging()
    
    # 在日志文件夹中保存配置文件副本
    config_copy_path = f"{log_dir}/config.json"
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("开始训练 MAPPO...")
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

    # 创建环境（用于获取维度信息）
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
    
    # 创建MAPPO智能体
    agent = MAPPOAgent(
        agent_state_dim=agent_state_dim,
        observe_dim=observe_dim,
        all_state_dim=all_state_dim,
        action_dim=action_dim,
        num_agents=env.kNumAgents,
        max_action=max_action,
        device=device,
        batch_size=config["training"]["batch_size"],
        gamma=config["training"]["gamma"],
        lam=config["training"]["lam"],
        lr=config["training"]["learning_rate"],
        eps_clip=config["training"]["eps_clip"],
        k_epochs=config["training"]["k_epochs"],
        entropy_coef=config["training"]["entropy_coef"],
        value_coef=config["training"]["value_coef"],
        max_grad_norm=config["training"]["max_grad_norm"]
    )
    
    logger.info("MAPPO智能体创建成功")
    
    # 创建多环境缓冲区
    num_envs = config["training"]["num_envs"]
    multi_buffer = MultiEnvMAPPOBuffer(
        num_envs=num_envs,
        max_episode_length=config["training"]["max_steps_per_episode"]
    )
    logger.info(f"多环境缓冲区创建成功 - 环境数量: {num_envs}")
    
    # 训练参数
    max_episodes = config["training"]["max_episodes"]
    max_steps = config["training"]["max_steps_per_episode"]
    save_interval = config["training"]["save_interval"]
    log_interval = config["training"]["log_interval"]
    update_interval = config["training"]["update_interval"]  # 每隔多少个episode更新一次
    
    # 记录训练过程的变量
    from collections import deque
    episode_rewards = deque(maxlen=log_interval * num_envs)
    episode_steps = deque(maxlen=log_interval * num_envs)
    success_episodes = deque(maxlen=log_interval * num_envs)
    completed_goals = deque(maxlen=log_interval * num_envs)
    episode_times = deque(maxlen=log_interval * num_envs)
    
    # 用于跟踪最佳平均完成任务数的变量
    best_avg_completed_goals = 0.0
    episodes_since_last_save = 0
    total_episodes_run = 0
    
    # 保存初始环境状态图
    logger.info("保存初始环境状态图...")
    try:
        if not os.path.exists('images'):
            os.makedirs('images')
        
        initial_plot_path = "images/initial_environment_state.png"
        env.plot_environment(save_path=initial_plot_path, fig_size=(12, 10), dpi=150)
        logger.info(f"初始环境状态图已保存到: {initial_plot_path}")
    except Exception as e:
        logger.warning(f"保存初始环境状态图失败: {str(e)}")
    
    logger.info("开始训练循环...")
    logger.info(f"使用{num_envs}个并行环境进行训练")
    start_time = time.time()
    
    for episode_batch in range(0, max_episodes, update_interval):
        batch_start_time = time.time()
        
        # 并行运行多个环境
        batch_results = []
        
        # 使用线程池并行运行环境
        with ThreadPoolExecutor(max_workers=min(num_envs, 8)) as executor:
            # 为每个环境提交任务
            future_to_env = {}
            for i in range(min(update_interval, max_episodes - episode_batch)):
                for env_id in range(num_envs):
                    future = executor.submit(
                        run_single_environment, 
                        env_id, config, agent, multi_buffer, max_steps
                    )
                    future_to_env[future] = (episode_batch + i, env_id)
            
            # 收集结果
            for future in as_completed(future_to_env):
                episode_idx, env_id = future_to_env[future]
                try:
                    result = future.get(timeout=300)  # 5分钟超时
                    batch_results.append(result)
                    total_episodes_run += 1
                    
                    # 记录统计信息
                    episode_rewards.append(result['reward'])
                    episode_steps.append(result['steps'])
                    success_episodes.append(result['success'])
                    completed_goals.append(result['completed_goals'])
                    episode_times.append(time.time() - batch_start_time)
                    
                except Exception as e:
                    logger.error(f"环境{env_id}运行失败: {str(e)}")
        
        # 使用收集到的经验更新网络
        ready_buffers = multi_buffer.get_ready_buffers()
        if ready_buffers:
            total_samples = multi_buffer.get_total_samples()
            logger.info(f"Episode batch {episode_batch//update_interval + 1}: 收集了{total_samples}个样本，开始更新网络...")
            
            # 对每个有数据的缓冲区进行更新
            update_stats = []
            for env_id, buffer in ready_buffers:
                if buffer.size > 0:
                    stats = agent.update(buffer)
                    update_stats.append(stats)
            
            # 清空所有缓冲区
            multi_buffer.clear_all()
            
            # 计算平均更新统计
            if update_stats:
                avg_stats = {}
                for key in update_stats[0].keys():
                    avg_stats[key] = np.mean([stat[key] for stat in update_stats])
                
                logger.info(f"网络更新完成 - Actor损失: {avg_stats['actor_loss']:.4f}, "
                           f"Critic损失: {avg_stats['critic_loss']:.4f}, "
                           f"熵: {avg_stats['entropy']:.4f}")
        
        episodes_since_last_save += len(batch_results)
        
        # 日志记录
        if total_episodes_run % (log_interval * num_envs) == 0 or episode_batch + update_interval >= max_episodes:
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                avg_steps = np.mean(episode_steps)
                success_rate = np.mean(success_episodes)
                avg_completed_goals = np.mean(completed_goals)
                avg_episode_time = np.mean(episode_times)
                
                logger.info(f"Episode {total_episodes_run}/{max_episodes * num_envs} - "
                           f"平均奖励: {avg_reward:.3f}, "
                           f"平均步数: {avg_steps:.1f}, "
                           f"成功率: {success_rate:.3f}, "
                           f"平均完成任务点: {avg_completed_goals:.2f}, "
                           f"平均用时: {avg_episode_time:.2f}s")
                
                # 检查是否需要保存最佳模型
                if avg_completed_goals > best_avg_completed_goals:
                    best_avg_completed_goals = avg_completed_goals
                    # 保存最佳模型
                    best_model_dir = f"{log_dir}/best_model"
                    agent.save_models(best_model_dir)
                    
                    logger.info(f"最佳模型已保存到 {best_model_dir} - 平均完成任务数提升至 {avg_completed_goals:.3f}")
                    episodes_since_last_save = 0
        
        # 定期保存
        if episodes_since_last_save >= save_interval:
            periodic_model_dir = f"{log_dir}/periodic_saves/episode_{total_episodes_run}"
            agent.save_models(periodic_model_dir)
            
            logger.info(f"定期模型已保存到 {periodic_model_dir}")
            episodes_since_last_save = 0
        
        # 检查是否完成训练
        if episode_batch + update_interval >= max_episodes:
            break
    
    total_time = time.time() - start_time
    logger.info(f"训练完成！总用时: {total_time:.2f}s")
    
    # 保存最终模型
    final_model_dir = f"{log_dir}/final_model"
    agent.save_models(final_model_dir)
    
    logger.info(f"最终模型已保存到 {final_model_dir}")
    
    # 输出最终训练统计
    if len(episode_rewards) > 0:
        final_avg_reward = np.mean(episode_rewards)
        final_avg_steps = np.mean(episode_steps)
        final_success_rate = np.mean(success_episodes)
        final_avg_completed_goals = np.mean(completed_goals)
        
        logger.info("=== 最终训练统计 ===")
        logger.info(f"最后{len(episode_rewards)}轮平均奖励: {final_avg_reward:.3f}")
        logger.info(f"最后{len(episode_steps)}轮平均步数: {final_avg_steps:.1f}")
        logger.info(f"最后{len(success_episodes)}轮成功率: {final_success_rate:.3f}")
        logger.info(f"最后{len(completed_goals)}轮平均完成任务点: {final_avg_completed_goals:.2f}")
        logger.info(f"历史最佳平均完成任务数: {best_avg_completed_goals:.3f}")
        logger.info(f"总训练轮数: {total_episodes_run}")
        logger.info(f"使用环境数量: {num_envs}")
        
        if len(episode_rewards) > 0:
            logger.info(f"最近轮次最高奖励: {np.max(episode_rewards):.3f}")
            logger.info(f"最近轮次最低奖励: {np.min(episode_rewards):.3f}")
        if len(completed_goals) > 0:
            logger.info(f"最近轮次最多完成任务点: {int(np.max(completed_goals))}")
            logger.info(f"最近轮次最少完成任务点: {int(np.min(completed_goals))}")
        logger.info("MAPPO训练结束！")

if __name__ == "__main__":
    train()
