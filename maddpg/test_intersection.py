import numpy as np
from enum import Enum

class RadarObserveType(Enum):
    Empty = 0
    Obstacle = 1
    Goal = 2
    Agent = 3
    Boundary = 4

class TestIntersection:
    def __init__(self):
        # 地图边界
        self.kMinX = 0.0
        self.kMaxX = 1.0
        self.kMinY = 0.0
        self.kMaxY = 1.0
        
        # 障碍物
        self.obstacles = np.array([[0.5, 0.5, 0.3]])  # 在(0.5, 0.5)处放置半径为0.3的障碍物

    def get_intersection(self, start_pos: np.ndarray, angle: float, max_dist: float):
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
            a = 1  # np.cos(angle)**2 + np.sin(angle)**2 = 1
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

        return min_dist, min_type

def test_intersection():
    # 创建测试实例
    test = TestIntersection()
    
    # 测试起点
    start_pos = np.array([0.1, 0.1])
    
    # 测试不同角度
    test_angles = [
        (-30, "测试-30度射线"),
        (0, "测试0度射线"),
        (30, "测试30度射线")
    ]
    
    for angle_deg, test_name in test_angles:
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle_deg)
        
        # 获取交点
        dist, obs_type = test.get_intersection(start_pos, angle_rad, 0.5)
        
        # 打印测试结果
        print(f"\n{test_name}:")
        print(f"角度: {angle_deg}度")
        print(f"检测到距离: {dist:.3f}")
        print(f"检测到类型: {RadarObserveType(obs_type).name}")
        
        # 验证结果
        # if angle_deg == -30:
        #     # -30度射线应该检测到障碍物
        #     assert obs_type == RadarObserveType.Obstacle.value
        #     assert 0.3 < dist < 0.5
        # elif angle_deg == 0:
        #     # 0度射线应该检测到右边界
        #     assert obs_type == RadarObserveType.Boundary.value
        #     assert abs(dist - 0.9) < 0.01
        # elif angle_deg == 30:
        #     # 30度射线应该检测到障碍物
        #     assert obs_type == RadarObserveType.Obstacle.value
        #     assert 0.3 < dist < 0.5

if __name__ == "__main__":
    test_intersection() 