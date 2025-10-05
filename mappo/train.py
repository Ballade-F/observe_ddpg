import json
import numpy as np
import torch
import logging
from datetime import datetime
from sim_env import SimEnv
import os
from replay_buffer import ReplayBuffer
from mappo_class import MAPPO


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
    config_path = './config.json'
    config = json.load(open(config_path, 'r'))
    
    # 设置日志
    logger, log_dir = setup_logging()
    models_dir = f"models/train_{datetime.now().strftime("%Y-%m-%d-%H-%M")}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
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


    # 参数设置
    actor_lr = 3e-4
    critic_lr = 1e-3
    total_episodes = 100000
    hidden_dim = 256
    gamma = 0.99
    lamda = 0.97
    eps = 0.3
    step_max = 600

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

    replay_buffer = ReplayBuffer(
        agent_n=env.kNumAgents,
        all_state_dim=all_state_dim,
        observe_state_dim=observe_dim,
        action_dim=action_dim,
        step_max=step_max
    )

    #记录指标
    ave_a_loss=[]
    ave_c_loss=[]
    ave_ent=[]
    ave_reward=[]
    ave_num_completed_goals=[]
    ave_step=[]

    for episode in range(1, total_episodes + 1):
        replay_buffer.reset_buffer()
        env.reset()

        current_global_state = get_global_state(env)
        current_agent_state = env.agentState
        current_observe_lengths = env.observeStateL
        current_observe_types = env.observeStateType

        #记录指标
        episode_reward=0

        for step in range(step_max):
            actions,a_logprob = mappo.choose_action(current_agent_state, current_observe_lengths, current_observe_types)
            next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions) 
            next_global_state = get_global_state(env)

            # 存储经验
            replay_buffer.store_transition(
                state=current_global_state,
                next_state=next_global_state,
                agent_state=current_agent_state,
                observe_l=current_observe_lengths,
                observe_t=current_observe_types,
                action=actions,
                a_logprob=a_logprob,
                reward=reward,
                done=done
            )

            # 更新状态
            current_global_state = next_global_state
            current_agent_state = next_agent_state
            current_observe_lengths = next_observe_lengths
            current_observe_types = next_observe_types

            #记录指标
            episode_reward += reward.sum()

            if done:
                break
        
        # 使用MAPPO更新参数
        a_loss, c_loss, ent = mappo.update(replay_buffer)

        # 记录指标,每10轮打印一次平均值
        num_completed_goals = np.sum(env.goalState[:, 2] > 0.5)
        ave_a_loss.append(a_loss)
        ave_c_loss.append(c_loss)
        ave_ent.append(ent)
        ave_reward.append(episode_reward)
        ave_step.append(step)
        ave_num_completed_goals.append(num_completed_goals)
        if episode % 10 == 0:
            logger.info(
                f"episode {episode}, actor loss: {np.mean(ave_a_loss)}, "
                f"critic loss: {np.mean(ave_c_loss)}, entropy: {np.mean(ave_ent)}, "
                f"reward: {np.mean(ave_reward)}, num completed goals: {np.mean(ave_num_completed_goals)}, "
                f"step: {np.mean(ave_step)}")
            ave_a_loss=[]
            ave_c_loss=[]
            ave_ent=[]
            ave_reward=[]
            ave_num_completed_goals=[]
            ave_step=[]

        if episode % 500 == 0:
            mappo.save_model(models_dir, episode)
            logger.info(f"Model saved at episode {episode}")



if __name__ == "__main__":
    train()