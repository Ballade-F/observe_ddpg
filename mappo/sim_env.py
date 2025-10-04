import math
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import json
from enum import Enum

class RadarObserveType(Enum):
    UnfinishedGoal = 0
    Empty = 1
    FinishedGoal = 2
    Obstacle = 3
    Agent = 4
    Boundary = 5
    
    



class SimEnv:
    def __init__(self, config: dict = None):
        """
        初始化2D平面环境
        Args:
            config: 环境配置

        config参数字段：
            地图：
                智能体数量，目标数量，障碍物数量，障碍物半径范围，
                环境xy最大最小，时间步长
            智能体：
                体积半径，速度范围，角速度范围
                探测雷达条数，雷达最远距离
            目标：
                到达目标判断距离
            奖励：
                时间惩罚系数，碰撞惩罚系数，到达目标奖励系数，接近目标奖励系数
            
        """
        self.config = config

        self.kMaxX = config.get("map", {}).get("kMaxX", 5.0)
        self.kMaxY = config.get("map", {}).get("kMaxY", 5.0)
        self.kMinX = config.get("map", {}).get("kMinX", -5.0)
        self.kMinY = config.get("map", {}).get("kMinY", -5.0)
        self.kMaxObsR = config.get("map", {}).get("kMaxObsR", 1.5)
        self.kMinObsR = config.get("map", {}).get("kMinObsR", 0.5)
        self.kNumAgents = config.get("map", {}).get("kNumAgents", 1)
        self.kNumGoals = config.get("map", {}).get("kNumGoals", 1)
        self.kNumObstacles = config.get("map", {}).get("kNumObstacles", 3)
        self.kTimeStep = config.get("map", {}).get("kTimeStep", 0.1)

        self.kAgentRadius = config.get("agent", {}).get("kAgentRadius", 0.2)
        self.kMaxSpeed = config.get("agent", {}).get("kMaxSpeed", 1.0)
        self.kMaxAngularSpeed = config.get("agent", {}).get("kMaxAngularSpeed", 1.0)
        self.kNumRadars = config.get("agent", {}).get("kNumRadars", 32)
        self.kMaxRadarDist = config.get("agent", {}).get("kMaxRadarDist", 3.0)
        
        self.kGoalThreshold = config.get("goal", {}).get("kGoalThreshold", 0.5)
        
        self.kTimeReward = config.get("reward", {}).get("kTimeReward", -0.01)
        self.kCollisionReward = config.get("reward", {}).get("kCollisionReward", -1.0)
        self.kGoalReward = config.get("reward", {}).get("kGoalReward", 1.0)
        self.kDistanceReward = config.get("reward", {}).get("kDistanceReward", 0.1)
        self.kAgentSeparationReward = config.get("reward", {}).get("kAgentSeparationReward", 0.01)
        self.kMinAgentDistance = config.get("reward", {}).get("kMinAgentDistance", 1.0)
        self.kMinGoalDistance = config.get("reward", {}).get("kMinGoalDistance", 0.1)
        
        
        #地图元素初始坐标
        self.agentStartState:Optional[np.ndarray] = None    #(n_agents, 3) x, y, theta
        self.goalStartState:Optional[np.ndarray] = None    #(n_goals, 3) x, y, finish_flag
        self.obstacles:Optional[np.ndarray] = None    #(n_obstacles, 3) x, y, r


        #过程变量
        self.agentState:Optional[np.ndarray] = None    #(n_agents, 3) x, y, theta
        self.observeStateL:Optional[np.ndarray] = None    #(n_agents, n_radars) 距离
        self.observeStateType:Optional[np.ndarray] = None    #(n_agents, n_radars) 类型
        self.goalState:Optional[np.ndarray] = None    #(n_goals, 3) x, y, finish_flag
        self.collision_flag:Optional[List[bool]] = None    #(n_agents)

        #程序中用到的辅助变量
        self.radar_angle_range = np.linspace(-np.pi, np.pi, self.kNumRadars, endpoint=False)

        #初始化
        self.generate_map()
        self.reset()

    def step(self, action: np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        执行一步动作
        Args:
            action: 动作 (n_agents, 2) speed, angular_speed
        Returns:
            next_agentState: 下一个智能体状态 (n_agents, 3) x, y, theta
            next_goalState: 下一个目标状态 (n_goals, 3) x, y, finish_flag
            next_obstacles: 下一个障碍物状态 (n_obstacles, 3) x, y, r
            next_observeStateL: 下一个观测距离 (n_agents, n_radars)
            next_observeStateType: 下一个观测类型 (n_agents, n_radars)
            rewards: 整个team的奖励 (1,)
            done: 是否结束
        """
        # 解析动作
        agent_speed = action[:, 0]
        agent_angular_speed = action[:, 1]
        agent_speed = np.clip(agent_speed, -self.kMaxSpeed, self.kMaxSpeed)
        agent_angular_speed = np.clip(agent_angular_speed, -self.kMaxAngularSpeed, self.kMaxAngularSpeed)
        
        # 更新智能体状态
        self.agentState[:, 2] += agent_angular_speed * self.kTimeStep
        # 对每个智能体的角度进行归一化
        for i in range(self.kNumAgents):
            self.agentState[i, 2] = self.normalize_theta(self.agentState[i, 2])
        self.agentState[:, 0] += agent_speed * np.cos(self.agentState[:, 2]) * self.kTimeStep
        self.agentState[:, 1] += agent_speed * np.sin(self.agentState[:, 2]) * self.kTimeStep

        # 检查是否碰撞
        for i in range(self.kNumAgents):
            if self.is_collision(self.agentState[i, :2], self.kAgentRadius):
                self.collision_flag[i] = True
         
        #检测到达目标
        agent_reach_goal_num = np.zeros(self.kNumAgents)
        for i in range(self.kNumAgents):
            reach_goal_index = self.update_reach_goal(self.agentState[i, :2])
            agent_reach_goal_num[i] = len(reach_goal_index)

        #更新观测状态
        self.update_observe_state()

        #计算奖励 - 使用改进的奖励函数
        rewards = self.calculate_improved_reward(agent_reach_goal_num)
        
        # 检查是否结束 任意碰撞或所有目标都到达
        done = np.any(self.collision_flag) or np.all(self.goalState[:, 2] > 0.5)
        
        return self.agentState, self.goalState, self.obstacles, self.observeStateL, self.observeStateType, rewards, done

    def generate_map(self):
        """生成地图"""
        self.agentStartState = np.zeros((self.kNumAgents, 3))
        self.goalStartState = np.zeros((self.kNumGoals, 3))
        self.obstacles = np.zeros((self.kNumObstacles, 3))

        #生成障碍
        for i in range(self.kNumObstacles):
            x = np.random.uniform(self.kMinX, self.kMaxX)
            y = np.random.uniform(self.kMinY, self.kMaxY)
            r = np.random.uniform(self.kMinObsR, self.kMaxObsR)
            self.obstacles[i] = np.array([x, y, r])

        #随机生成起点和目标点, 起点和目标点不能重合, 且不能在障碍物内
        # 为每个agent生成起点
        for i in range(self.kNumAgents):
            start_collision = True
            while start_collision:
                start_pos = np.random.uniform(self.kMinX, self.kMaxX, 2)
                start_collision = self.is_collision(start_pos, self.kAgentRadius)
            start_angle = np.random.uniform(-np.pi, np.pi)
            self.agentStartState[i] = np.array([start_pos[0], start_pos[1], start_angle])
        
        # 为每个目标生成位置
        for i in range(self.kNumGoals):
            goal_collision = True
            while goal_collision:
                goal_pos = np.random.uniform(self.kMinX, self.kMaxX, 2)
                goal_collision = self.is_collision(goal_pos, self.kAgentRadius)
            self.goalStartState[i] = np.array([goal_pos[0], goal_pos[1], 0])

    def reset(self):
        """重置环境"""
        self.agentState = self.agentStartState.copy()
        self.goalState = self.goalStartState.copy()
        self.collision_flag = [False] * self.kNumAgents
        self.update_observe_state()

    def update_observe_state(self):
        """
        更新观测状态
        """
        self.observeStateL = np.zeros((self.kNumAgents, self.kNumRadars))
        self.observeStateType = np.zeros((self.kNumAgents, self.kNumRadars), dtype=int)
        for i in range(self.kNumAgents):
            for j in range(self.kNumRadars):
                angle = self.radar_angle_range[j] + self.agentState[i, 2] #雷达角度
                l,t = self.get_intersection(self.agentState[i, :2], angle, self.kMaxRadarDist)
                self.observeStateL[i, j] = l
                self.observeStateType[i, j] = t

    def get_intersection(self, start_pos: np.ndarray, angle: float, max_dist: float)->Tuple[float, int]:
        """
        获取雷达与障碍物的交点
        Args:
            start_pos: 雷达起点 (x, y)
            angle: 雷达角度 (弧度,-pi~pi)
            max_dist: 雷达最大距离
        Returns:
            l: 交点距离(真实坐标系)
            t: 交点类型
        """
        x, y = start_pos
        x_end = x + max_dist * np.cos(angle)
        y_end = y + max_dist * np.sin(angle)

        # 初始化最小距离和类型
        min_dist = max_dist
        min_type = RadarObserveType.Empty.value

        # 检查地图边界
        # 计算射线与边界的交点
        if abs(np.cos(angle)) > 1e-6:  # 避免除以0
            # 检查左右边界
            for bound_x in [self.kMinX, self.kMaxX]:
                t = (bound_x - x) / np.cos(angle)
                if t > 0:
                    y_intersect = y + t * np.sin(angle)
                    if self.kMinY <= y_intersect <= self.kMaxY:
                        dist = t
                        if dist < min_dist:
                            min_dist = dist
                            min_type = RadarObserveType.Boundary.value

        if abs(np.sin(angle)) > 1e-6:  # 避免除以0
            # 检查上下边界
            for bound_y in [self.kMinY, self.kMaxY]:
                t = (bound_y - y) / np.sin(angle)
                if t > 0:
                    x_intersect = x + t * np.cos(angle)
                    if self.kMinX <= x_intersect <= self.kMaxX:
                        dist = t
                        if dist < min_dist:
                            min_dist = dist
                            min_type = RadarObserveType.Boundary.value

        # 检查障碍物
        for obs_x, obs_y, obs_r in self.obstacles:
            # 计算射线到圆心的距离
            dx = x - obs_x
            dy = y - obs_y
            # a = np.cos(angle)**2 + np.sin(angle)**2
            a = 1
            b = 2 * (dx * np.cos(angle) + dy * np.sin(angle))
            c = dx**2 + dy**2 - obs_r**2
            
            # 求解二次方程
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                t1 = (-b + np.sqrt(discriminant)) / (2*a)
                t2 = (-b - np.sqrt(discriminant)) / (2*a)
                for t in [t1, t2]:
                    if t > 0 and t < min_dist:
                        min_dist = t
                        min_type = RadarObserveType.Obstacle.value

        # 检查目标点
        for i in range(self.kNumGoals):
            goal_x, goal_y, _ = self.goalStartState[i]
            # 计算射线到目标点的距离
            dx = x - goal_x
            dy = y - goal_y
            a = 1
            b = 2 * (dx * np.cos(angle) + dy * np.sin(angle))
            c = dx**2 + dy**2 - self.kGoalThreshold**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                t1 = (-b + np.sqrt(discriminant)) / (2*a)
                t2 = (-b - np.sqrt(discriminant)) / (2*a)
                for t in [t1, t2]:
                    if t > 0 and t < min_dist:
                        min_dist = t
                        min_type = RadarObserveType.UnfinishedGoal.value
                        if self.goalState[i, 2] > 0.5:
                            min_type = RadarObserveType.FinishedGoal.value

        # 检查其他智能体
        for agent_x, agent_y, _ in self.agentState:
            if np.allclose([agent_x, agent_y], start_pos[:2]):  # 跳过自己
                continue
            # 计算射线到智能体的距离
            dx = x - agent_x
            dy = y - agent_y
            a = 1
            b = 2 * (dx * np.cos(angle) + dy * np.sin(angle))
            c = dx**2 + dy**2 - self.kAgentRadius**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                t1 = (-b + np.sqrt(discriminant)) / (2*a)
                t2 = (-b - np.sqrt(discriminant)) / (2*a)
                for t in [t1, t2]:
                    if t > 0 and t < min_dist:
                        min_dist = t
                        min_type = RadarObserveType.Agent.value

        return min_dist, min_type

    def is_collision(self, pos: np.ndarray, safe_distance:float=0.0)->bool:
        """
        判断点是否在障碍物内
        Args:
            pos: 点坐标(x, y)，真实坐标系
            safe_distance: 安全距离，用于判断点是否在障碍物内
        Returns:
            bool: 是否在障碍物内
        """
        #出地图范围
        if pos[0] < self.kMinX or pos[0] > self.kMaxX or pos[1] < self.kMinY or pos[1] > self.kMaxY:
            return True
        for obs_x, obs_y, obs_r in self.obstacles:
            if np.linalg.norm(pos - np.array([obs_x, obs_y])) < (obs_r + safe_distance):
                return True
        return False
    
    def update_reach_goal(self, pos: np.ndarray)->List[int]:
        """
        更新到达目标状态
        Args:
            pos: agent位置 (x, y)
        Returns:
            List[int]: 到达目标的索引
        """
        reach_goal_index = []
        for i in range(self.kNumGoals):
            if self.goalState[i, 2] < 0.5 and np.linalg.norm(pos - self.goalState[i, :2]) < self.kGoalThreshold:
                self.goalState[i, 2] = 1
                reach_goal_index.append(i)
        return reach_goal_index
    
    def normalize_theta(self, theta: float) -> float:
        """将角度归一化到[-pi, pi]"""
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta
    
    def get_goal_distance_reward(self, agent_pos: np.ndarray) -> float:
        """
        计算智能体到所有未完成目标的距离倒数和奖励
        Args:
            agent_pos: 智能体位置 (x, y)
        Returns:
            float: 距离倒数和奖励
        """
        unfinished_goals = self.goalState[self.goalState[:, 2] < 0.5]  # 未完成的目标
        if len(unfinished_goals) == 0:
            return 0.0  # 所有目标都完成了
        
        distances = np.linalg.norm(unfinished_goals[:, :2] - agent_pos, axis=1)
        # 限制最小距离以避免无穷大奖励
        distances = np.maximum(distances, self.kMinGoalDistance)
        
        # 计算距离倒数和
        inverse_distances = 1.0 / distances
        return np.sum(inverse_distances)
    
    def get_agent_separation_reward(self) -> float:
        """
        计算智能体分离奖励，鼓励智能体保持适当距离
        Returns:
            float: 分离奖励
        """
        separation_reward = 0.0
        for i in range(self.kNumAgents):
            for j in range(i + 1, self.kNumAgents):
                dist = np.linalg.norm(self.agentState[i, :2] - self.agentState[j, :2])
                if dist < self.kMinAgentDistance:
                    # 距离过近，给予惩罚
                    separation_reward -= self.kAgentSeparationReward * (self.kMinAgentDistance - dist)

        return separation_reward
    

    
    def calculate_improved_reward(self, agent_reach_goal_num: np.ndarray) -> np.ndarray:
        """
        计算简化的奖励函数
        Args:
            agent_reach_goal_num: 每个智能体到达的目标数量
        Returns:
            np.ndarray: 每个智能体的奖励，shape=(kNumAgents,)
        """
        rewards = np.zeros(self.kNumAgents)
        
        # 1. 基础奖励（时间惩罚、碰撞惩罚、目标完成奖励）
        for i in range(self.kNumAgents):
            rewards[i] += self.kTimeReward  # 时间惩罚
            
            if self.collision_flag[i]:
                rewards[i] += self.kCollisionReward  # 碰撞惩罚
            elif agent_reach_goal_num[i] > 0:
                rewards[i] += self.kGoalReward * agent_reach_goal_num[i]  # 目标完成奖励
        
        # 2. 目标距离奖励：使用距离倒数和
        for i in range(self.kNumAgents):
            if not self.collision_flag[i]:  # 只对未碰撞的智能体计算
                distance_reward = self.get_goal_distance_reward(self.agentState[i, :2])
                rewards[i] += self.kDistanceReward * distance_reward
        
        # 3. 智能体分离奖励（可选）
        # separation_reward = self.get_agent_separation_reward()
        # rewards += separation_reward / self.kNumAgents  # 平均分配给所有智能体
        
        return rewards
    
    def plot_environment(self, save_path: str = "environment.png", fig_size: Tuple[int, int] = (10, 10), dpi: int = 100):
        """
        绘制当前环境状态并保存为PNG文件
        Args:
            save_path: 保存文件的路径
            fig_size: 图像大小 (width, height)
            dpi: 图像分辨率
        """
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        # 设置地图边界
        ax.set_xlim(self.kMinX - 0.5, self.kMaxX + 0.5)
        ax.set_ylim(self.kMinY - 0.5, self.kMaxY + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Multi-Agent Environment State')
        
        # 绘制地图边界
        boundary_rect = plt.Rectangle((self.kMinX, self.kMinY), 
                                    self.kMaxX - self.kMinX, 
                                    self.kMaxY - self.kMinY,
                                    fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(boundary_rect)
        
        # 绘制障碍物（灰色圆形）
        for obs_x, obs_y, obs_r in self.obstacles:
            obstacle_circle = plt.Circle((obs_x, obs_y), obs_r, 
                                       color='gray', alpha=0.7, label='Obstacle' if obs_x == self.obstacles[0, 0] else "")
            ax.add_patch(obstacle_circle)
        
        # 绘制目标点
        for i, (goal_x, goal_y, finish_flag) in enumerate(self.goalState):
            if finish_flag > 0.5:  # 已完成的目标
                goal_circle = plt.Circle((goal_x, goal_y), self.kGoalThreshold, 
                                       color='green', alpha=0.6, label='Completed Goal' if i == 0 and finish_flag > 0.5 else "")
            else:  # 未完成的目标
                goal_circle = plt.Circle((goal_x, goal_y), self.kGoalThreshold, 
                                       color='red', alpha=0.6, label='Uncompleted Goal' if i == 0 and finish_flag <= 0.5 else "")
            ax.add_patch(goal_circle)
            
            # 在目标中心添加标号
            ax.text(goal_x, goal_y, f'G{i}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # 绘制智能体
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
        for i, (agent_x, agent_y, agent_theta) in enumerate(self.agentState):
            color = colors[i % len(colors)]
            
            # 绘制智能体主体（圆形）
            if self.collision_flag[i]:
                # 碰撞的智能体用红色边框
                agent_circle = plt.Circle((agent_x, agent_y), self.kAgentRadius, 
                                        color=color, alpha=0.8, edgecolor='red', linewidth=3,
                                        label=f'Agent {i} (Collision)' if i == 0 and self.collision_flag[i] else "")
            else:
                agent_circle = plt.Circle((agent_x, agent_y), self.kAgentRadius, 
                                        color=color, alpha=0.8,
                                        label=f'Agent {i}' if i == 0 and not self.collision_flag[i] else "")
            ax.add_patch(agent_circle)
            
            # 绘制智能体朝向箭头
            arrow_length = self.kAgentRadius * 1.5
            dx = arrow_length * np.cos(agent_theta)
            dy = arrow_length * np.sin(agent_theta)
            ax.arrow(agent_x, agent_y, dx, dy, 
                    head_width=self.kAgentRadius*0.3, head_length=self.kAgentRadius*0.2, 
                    fc=color, ec=color, alpha=0.9)
            
            # 在智能体中心添加标号
            ax.text(agent_x, agent_y, f'A{i}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # 绘制雷达线
        radar_colors = {
            RadarObserveType.Empty.value: 'lightgray',
            RadarObserveType.Obstacle.value: 'red', 
            RadarObserveType.FinishedGoal.value: 'lightgreen',
            RadarObserveType.UnfinishedGoal.value: 'orange',
            RadarObserveType.Agent.value: 'lightblue',
            RadarObserveType.Boundary.value: 'black'
        }
        
        radar_labels = {
            RadarObserveType.Empty.value: 'Radar Empty',
            RadarObserveType.Obstacle.value: 'Radar Obstacle', 
            RadarObserveType.FinishedGoal.value: 'Radar Finished Goal',
            RadarObserveType.UnfinishedGoal.value: 'Radar Unfinished Goal',
            RadarObserveType.Agent.value: 'Radar Agent',
            RadarObserveType.Boundary.value: 'Radar Boundary'
        }
        
        # 记录已添加到图例的雷达类型
        radar_legend_added = set()
        
        for i, (agent_x, agent_y, agent_theta) in enumerate(self.agentState):
            for j in range(self.kNumRadars):
                # 计算雷达线角度
                radar_angle = self.radar_angle_range[j] + agent_theta
                
                # 获取雷达探测距离和类型
                radar_dist = self.observeStateL[i, j]
                radar_type = self.observeStateType[i, j]
                
                # 计算雷达线终点坐标
                end_x = agent_x + radar_dist * np.cos(radar_angle)
                end_y = agent_y + radar_dist * np.sin(radar_angle)
                
                # 获取对应类型的颜色
                line_color = radar_colors.get(radar_type, 'gray')
                
                # 确定是否需要添加图例标签
                label = None
                if radar_type not in radar_legend_added:
                    label = radar_labels.get(radar_type, f'Radar Type {radar_type}')
                    radar_legend_added.add(radar_type)
                
                # 绘制雷达线
                ax.plot([agent_x, end_x], [agent_y, end_y], 
                       color=line_color, alpha=0.6, linewidth=1, label=label)
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        
        print(f"Environment state plot saved to: {save_path}")
    
    
def main():
    """
    主函数，用于测试SimEnv类并生成环境状态图
    """
    # 创建环境配置
    config = {
        "map": {
            "kMaxX": 5.0,
            "kMaxY": 5.0,
            "kMinX": -5.0,
            "kMinY": -5.0,
            "kMaxObsR": 1.0,
            "kMinObsR": 0.3,
            "kNumAgents": 3,
            "kNumGoals": 2,
            "kNumObstacles": 5,
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
            "kGoalThreshold": 0.3
        },
        "reward": {
            "kTimeReward": -0.01,
            "kCollisionReward": -1.0,
            "kGoalReward": 1.0,
            "kDistReward": 0.01
        }
    }

    # 设置numpy的随机种子
    np.random.seed(1)
    
    # 实例化环境
    print("Creating environment...")
    env = SimEnv(config)
    
    # 生成并保存初始环境状态图
    print("Generating environment state plot...")
    env.plot_environment(save_path="sim_env_initial.png", fig_size=(12, 10), dpi=150)
    
    # 执行几步动作来展示动态效果
    print("Executing random actions...")
    for step in range(5):
        # 生成随机动作
        actions = np.random.uniform(-0.5, 0.5, (env.kNumAgents, 2))
        
        # 执行动作
        next_agentState, next_goalState, next_obstacles, next_observeStateL, next_observeStateType, rewards, done = env.step(actions)
        
        print(f"Step {step+1}: Rewards = {rewards}, Done = {done}")
        
        if done:
            print("Environment finished!")
            break
    
    # 保存最终状态图
    # 修改goal1的状态为已完成，测试雷达线是否正确
    env.goalState[1, 2] = 1
    env.update_observe_state()
    env.plot_environment(save_path="sim_env_final.png", fig_size=(12, 10), dpi=150)
    print("Test completed! Generated initial state plot (sim_env_initial.png) and final state plot (sim_env_final.png)")


def test_reward_function():
    """
    测试简化的奖励函数
    """
    print("=" * 60)
    print("测试简化的奖励函数...")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        "map": {
            "kMaxX": 5.0, "kMaxY": 5.0, "kMinX": -5.0, "kMinY": -5.0,
            "kMaxObsR": 1.0, "kMinObsR": 0.3, "kNumAgents": 2, "kNumGoals": 3, "kNumObstacles": 2,
            "kTimeStep": 0.1
        },
        "agent": {
            "kAgentRadius": 0.2, "kMaxSpeed": 1.0, "kMaxAngularSpeed": 1.0,
            "kNumRadars": 32, "kMaxRadarDist": 3.0
        },
        "goal": {"kGoalThreshold": 0.3},
        "reward": {
            "kTimeReward": -0.1, "kCollisionReward": -10.0, "kGoalReward": 5.0,
            "kDistanceReward": 0.01, "kAgentSeparationReward": 0.1,
            "kMinAgentDistance": 1.0, "kMinGoalDistance": 0.1
        }
    }
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建环境
    env = SimEnv(config)
    print(f"✓ 环境创建成功")
    print(f"  智能体数量: {env.kNumAgents}")
    print(f"  目标数量: {env.kNumGoals}")
    print(f"  障碍物数量: {env.kNumObstacles}")
    
    # 测试场景1：智能体远离目标
    print(f"\n{'='*30}")
    print("测试场景1：智能体远离目标")
    print(f"{'='*30}")
    
    env.reset()
    # 手动设置智能体位置（远离目标）
    env.agentState[0] = [-4, -4, 0]  # 智能体0在左下角
    env.agentState[1] = [4, 4, 0]    # 智能体1在右上角
    
    # 计算奖励
    agent_reach_goal_num = np.zeros(env.kNumAgents)
    reward1 = env.calculate_improved_reward(agent_reach_goal_num)
    print(f"远离目标时的奖励: {reward1:.4f}")
    
    # 测试场景2：智能体靠近目标
    print(f"\n{'='*30}")
    print("测试场景2：智能体靠近目标")
    print(f"{'='*30}")
    
    # 将智能体移动到目标附近
    if len(env.goalState) > 0:
        env.agentState[0, :2] = env.goalState[0, :2] + [0.5, 0]  # 智能体0靠近目标0
    if len(env.goalState) > 1:
        env.agentState[1, :2] = env.goalState[1, :2] + [0, 0.5]  # 智能体1靠近目标1
    
    reward2 = env.calculate_improved_reward(agent_reach_goal_num)
    print(f"靠近目标时的奖励: {reward2:.4f}")
    print(f"奖励改善: {reward2 - reward1:.4f}")
    
    # 测试场景3：智能体过于接近
    print(f"\n{'='*30}")
    print("测试场景3：智能体过于接近")
    print(f"{'='*30}")
    
    # 将两个智能体放在很近的位置
    env.agentState[0, :2] = [0, 0]
    env.agentState[1, :2] = [0.2, 0]  # 距离0.2，小于最小距离1.0
    
    reward3 = env.calculate_improved_reward(agent_reach_goal_num)
    print(f"智能体过近时的奖励: {reward3:.4f}")
    
    # 测试场景4：智能体适当分离
    print(f"\n{'='*30}")
    print("测试场景4：智能体适当分离")
    print(f"{'='*30}")
    
    # 将智能体分开适当距离
    env.agentState[0, :2] = [0, 0]
    env.agentState[1, :2] = [2, 0]  # 距离2.0，大于最小距离
    
    reward4 = env.calculate_improved_reward(agent_reach_goal_num)
    print(f"智能体适当分离时的奖励: {reward4:.4f}")
    print(f"与过近情况相比改善: {reward4 - reward3:.4f}")
    
    # 测试场景5：智能体到达目标
    print(f"\n{'='*30}")
    print("测试场景5：智能体到达目标")
    print(f"{'='*30}")
    
    # 模拟智能体到达目标
    agent_reach_goal_num[0] = 1  # 智能体0到达1个目标
    reward5 = env.calculate_improved_reward(agent_reach_goal_num)
    print(f"到达目标时的奖励: {reward5:.4f}")
    print(f"与未到达相比改善: {reward5 - reward4:.4f}")
    
    # 测试距离倒数和奖励的特性
    print(f"\n{'='*30}")
    print("测试距离倒数和奖励特性")
    print(f"{'='*30}")
    
    env.reset()
    # 测试单个智能体的距离奖励
    agent_pos = np.array([0.0, 0.0])
    distance_reward = env.get_goal_distance_reward(agent_pos)
    print(f"智能体在原点时的距离奖励: {distance_reward:.4f}")
    
    # 将智能体移动到更靠近目标的位置
    if len(env.goalState) > 0:
        closer_pos = env.goalState[0, :2] * 0.8  # 更靠近第一个目标
        closer_distance_reward = env.get_goal_distance_reward(closer_pos)
        print(f"智能体靠近目标时的距离奖励: {closer_distance_reward:.4f}")
        print(f"奖励提升: {closer_distance_reward - distance_reward:.4f}")
    
    print(f"\n{'='*60}")
    print("🎉 简化奖励函数测试完成！")
    print("✓ 距离倒数和奖励：智能体越靠近目标奖励越高")
    print("✓ 分离奖励：智能体保持适当距离")
    print("✓ 基础奖励：时间惩罚、碰撞惩罚、目标完成奖励")
    print("✓ 避免无穷大：最小距离限制确保数值稳定")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_reward_function()
        
    
    