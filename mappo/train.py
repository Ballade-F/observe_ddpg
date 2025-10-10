import json
import numpy as np
import torch
import logging
from datetime import datetime
from sim_env import SimEnv
import os
from replay_buffer import ReplayBuffer, ReplayBufferBatch
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
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建多个环境以提高泛化能力
    num_envs = 10
    envs = []
    logger.info(f"开始创建 {num_envs} 个训练环境（每个环境配置不同）...")
    
    for env_idx in range(num_envs):
        # 连续创建环境，随机数生成器状态会自动变化，确保每个环境配置不同
        env = SimEnv(config)
        envs.append(env)
        logger.info(f"环境 {env_idx+1}/{num_envs} 创建成功 - Agent数量: {env.kNumAgents}, 目标数量: {env.kNumGoals}, 障碍物数量: {env.kNumObstacles}")
    
    # 使用第一个环境获取维度信息
    env_template = envs[0]
    logger.info(f"所有环境创建完成，将在训练时轮流使用以提高泛化能力")
    
    # 计算状态和动作维度
    agent_state_dim = 3  # x, y, theta
    observe_dim = env_template.kNumRadars  # 雷达观测维度
    action_dim = 2  # speed, angular_speed
    all_state_dim = env_template.kNumAgents * 3 + env_template.kNumGoals * 3 + env_template.kNumObstacles * 3
    max_action = np.array([env_template.kMaxSpeed, env_template.kMaxAngularSpeed])
    
    logger.info(f"状态维度 - Agent状态: {agent_state_dim}, 观测维度: {observe_dim}, 动作维度: {action_dim}")
    logger.info(f"全局状态维度: {all_state_dim}, 最大动作: {max_action}")


    # 参数设置
    actor_lr = config["training"]["actor_lr"]
    critic_lr = config["training"]["critic_lr"]
    total_episodes = config["training"]["total_episodes"]
    hidden_dim = config["network"]["hidden_dim"]
    gamma = config["training"]["gamma"]
    lamda = config["training"]["lamda"]
    eps = config["training"]["epsilon"]
    step_max = config["training"]["step_max"]

    mappo = MAPPO(
        agent_n=env_template.kNumAgents,
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
        agent_n=env_template.kNumAgents,
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
        # 轮流选择环境进行训练，以提高泛化能力
        env_idx = (episode - 1) % num_envs
        env = envs[env_idx]
        
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

        if episode % 100 == 0:
            mappo.save_model(models_dir, episode)
            logger.info(f"Model saved at episode {episode}")



def train_batch():
    """批量训练函数 - 使用ReplayBufferBatch对多个环境进行批量训练"""
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
    
    logger.info("开始批量训练 MAPPO...")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"配置文件已保存到: {config_copy_path}")
    logger.info(f"配置参数: {config}")
    
    # 设置随机种子
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建多个环境以提高泛化能力
    num_envs = 10
    envs = []
    logger.info(f"开始创建 {num_envs} 个训练环境（每个环境配置不同）...")
    
    for env_idx in range(num_envs):
        # 连续创建环境，随机数生成器状态会自动变化，确保每个环境配置不同
        env = SimEnv(config)
        envs.append(env)
        logger.info(f"环境 {env_idx+1}/{num_envs} 创建成功 - Agent数量: {env.kNumAgents}, 目标数量: {env.kNumGoals}, 障碍物数量: {env.kNumObstacles}")
    
    # 使用第一个环境获取维度信息
    env_template = envs[0]
    logger.info(f"所有环境创建完成，将进行批量训练")
    
    # 计算状态和动作维度
    agent_state_dim = 3  # x, y, theta
    observe_dim = env_template.kNumRadars  # 雷达观测维度
    action_dim = 2  # speed, angular_speed
    all_state_dim = env_template.kNumAgents * 3 + env_template.kNumGoals * 3 + env_template.kNumObstacles * 3
    max_action = np.array([env_template.kMaxSpeed, env_template.kMaxAngularSpeed])
    
    logger.info(f"状态维度 - Agent状态: {agent_state_dim}, 观测维度: {observe_dim}, 动作维度: {action_dim}")
    logger.info(f"全局状态维度: {all_state_dim}, 最大动作: {max_action}")

    # 参数设置
    actor_lr = config["training"]["actor_lr"]
    critic_lr = config["training"]["critic_lr"]
    total_episodes = config["training"]["total_episodes"]
    hidden_dim = config["network"]["hidden_dim"]
    gamma = config["training"]["gamma"]
    lamda = config["training"]["lamda"]
    eps = config["training"]["epsilon"]
    step_max = config["training"]["step_max"]

    mappo = MAPPO(
        agent_n=env_template.kNumAgents,
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

    # 使用ReplayBufferBatch，batch_size为num_envs
    replay_buffer_batch = ReplayBufferBatch(
        agent_n=env_template.kNumAgents,
        all_state_dim=all_state_dim,
        observe_state_dim=observe_dim,
        action_dim=action_dim,
        batch_size=num_envs,
        step_max=step_max
    )

    # 记录指标（累积num_envs轮的指标）
    batch_a_loss = []
    batch_c_loss = []
    batch_ent = []
    batch_reward = []
    batch_num_completed_goals = []
    batch_step = []

    episode = 0
    while episode < total_episodes:
        # 重置batch buffer
        replay_buffer_batch.reset_buffer()
        
        # 对num_envs个环境分别进行rollout
        for traj_idx in range(num_envs):
            if episode >= total_episodes:
                break
                
            episode += 1
            env = envs[traj_idx]
            
            # 重置环境
            env.reset()
            
            current_global_state = get_global_state(env)
            current_agent_state = env.agentState
            current_observe_lengths = env.observeStateL
            current_observe_types = env.observeStateType

            # 记录该episode的指标
            episode_reward = 0

            for step in range(step_max):
                actions, a_logprob = mappo.choose_action(current_agent_state, current_observe_lengths, current_observe_types)
                next_agent_state, next_goal_state, next_obstacles, next_observe_lengths, next_observe_types, reward, done = env.step(actions)
                next_global_state = get_global_state(env)

                # 存储经验到batch buffer的对应轨迹
                replay_buffer_batch.store_transition(
                    traj_idx=traj_idx,
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

                # 记录指标
                episode_reward += reward.sum()

                if done:
                    break
            
            # 记录该轨迹的指标
            num_completed_goals = np.sum(env.goalState[:, 2] > 0.5)
            batch_reward.append(episode_reward)
            batch_step.append(step)
            batch_num_completed_goals.append(num_completed_goals)
        
        # 对整个batch进行一次更新
        a_loss, c_loss, ent = mappo.update_batch(replay_buffer_batch)
        batch_a_loss.append(a_loss)
        batch_c_loss.append(c_loss)
        batch_ent.append(ent)
        
        # 每次update_batch后打印日志
        logger.info(
            f"episode {episode}, actor loss: {np.mean(batch_a_loss)}, "
            f"critic loss: {np.mean(batch_c_loss)}, entropy: {np.mean(batch_ent)}, "
            f"reward: {np.mean(batch_reward)}, num completed goals: {np.mean(batch_num_completed_goals)}, "
            f"step: {np.mean(batch_step)}")
        
        # 清空批次指标
        batch_a_loss = []
        batch_c_loss = []
        batch_ent = []
        batch_reward = []
        batch_num_completed_goals = []
        batch_step = []

        # 每100个episode保存一次模型
        if episode % 100 == 0:
            mappo.save_model(models_dir, episode)
            logger.info(f"Model saved at episode {episode}")


if __name__ == "__main__":
    # train()  # 原始单轨迹训练
    train_batch()  # 批量训练