import numpy as np

class CustomEVRPInstance:
    def __init__(self, locations, vehicles):
        self.locations = locations  # 保存所有位置数据
        self.vehicles = vehicles  # 保存所有车辆数据

        # 创建一个位置 ID 到索引的映射，以便快速查找位置索引
        self.location_id_to_index = {loc['id']: idx for idx, loc in enumerate(locations)}

        # 基础参数定义
        self.n = sum(1 for loc in locations if loc['type'] == 'c')  # 客户数量
        self.m = sum(1 for loc in locations if loc['type'] == 'f')  # 充电站数量
        self.O = next(loc for loc in locations if loc['type'] == 'd')  # 配送中心起点
        self.O_prime = self.n + self.m + 1  # 配送中心终点
        self.N = self.n + self.m + 2  # 所有节点数量，包括起点和终点

        self.Q_e = vehicles['fuel_tank_capacity']  # 电动车载重量
        self.Q_f = 150  # 燃油车载重量（假设值，可以根据需要调整）
        self.B_star = 200  # 电动车电池额定电量
        self.v = vehicles['average_velocity']  # 车辆行驶速度
        self.m_v = 2000  # 车辆自重
        self.e = vehicles['inverse_refueling_rate']  # 车辆充电速度，单位时间充电量

        self.p_1 = 500  # 电动车单位固定成本
        self.p_2 = 400  # 燃油车单位固定成本
        self.p_3 = 2  # 电动车单位运输成本
        self.p_4 = 3  # 燃油车单位运输成本
        self.p_5 = 10  # 单位重量的价格
        self.theta_1 = 0.01  # 运输与补电时的产品新鲜度衰减系数
        self.theta_2 = 0.02  # 卸货时的产品新鲜度衰减系数
        self.c = 0.5  # 碳税价格
        self.p_6 = 0.1  # 电动车单位充电成本
        self.p_7 = 50  # 车辆早到惩罚成本
        self.p_8 = 50  # 车辆迟到惩罚成本
        self.soc_min = 0.2  # 电动车最低电量负荷比
        self.M = 1e6  # 一个极大正值

        # 客户点需求量和时间窗
        self.customer_demand = np.array([loc['demand'] for loc in locations if loc['type'] == 'c'])
        self.time_window_start = np.array([loc['ready_time'] for loc in locations if loc['type'] == 'c'])
        self.time_window_end = self.time_window_start + np.array([loc['due_date'] for loc in locations if loc['type'] == 'c'])

        # 距离矩阵和行驶时间矩阵
        coords = np.array([(loc['x'], loc['y']) for loc in locations])
        self.distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)  # 使用欧氏距离计算距离矩阵
        self.travel_time_matrix = self.distance_matrix / self.v  # 根据车辆速度计算行驶时间矩阵

        # 车辆数量
        self.k_e = 2  # 电动车数量（假设值，可以根据需要调整）
        self.k_f = 3  # 燃油车数量（假设值，可以根据需要调整）

        # 电动车能耗计算
        c_d = 0.3  # 空气阻力系数
        rho = 1.225  # 空气密度 (kg/m^3)
        A = 2.5  # 车辆迎风面积 (m^2)
        g = 9.81  # 重力加速度 (m/s^2)
        phi_d = 0.9  # 电池放电效率
        varphi_d = 0.85  # 电机效率

        # 电动车能耗系数计算公式：
        # 公式的前半部分(0.5 * c_d * rho * A * v^3)表示空气阻力功率，
        # 后半部分((m_v + 负载) * g * c_d * v)表示克服滚动阻力的功率。
        K_ijk = 0.5 * c_d * rho * A * self.v ** 3 + (self.m_v + self.customer_demand.mean()) * g * c_d * self.v
        self.L_ijk = phi_d * varphi_d * K_ijk * self.travel_time_matrix  # 乘以行驶时间，计算总能耗

        # 燃油车油耗计算
        xi = 14.7  # 燃油空气质量比
        kappa = 44.8  # 燃油热值 (kJ/g)
        psi = 0.85  # 燃油转换系数
        sigma = 0.5  # 发动机摩擦系数 (kJ/r/L)
        vartheta = 30  # 发动机转速 (r/s)
        omega = 2.0  # 发动机排量 (L)
        eta = 0.3  # 燃油机效率
        tau = 0.9  # 车传动系统效率

        # 燃油车油耗系数计算公式：
        # 公式的前半部分(xi / (kappa * psi))表示单位燃油的能量转换效率，
        # 后半部分(sigma * vartheta * omega + K_ijk / (eta * tau))表示克服发动机摩擦和车辆滚动阻力的功率。
        self.F_ijk = (xi / (kappa * psi)) * (sigma * vartheta * omega + K_ijk / (eta * tau))

        # 碳排放量计算
        pi_e = 2.31  # 燃油排放系数 (kg CO2/L)
        self.E_ijk = pi_e * self.F_ijk  # 乘以燃油消耗，得到总碳排放量

        # 固定成本
        self.fixed_cost = self.p_1 * self.k_e + self.p_2 * self.k_f  # 电动车和燃油车的总固定成本

        # 运输成本
        self.transport_cost = self.p_3 * np.sum(self.distance_matrix) + self.p_4 * np.sum(self.distance_matrix)  # 所有车辆的运输成本

        # 货损成本
        decay_transport = lambda t: 1 - np.exp(-self.theta_1 * t)  # 运输与补电时的衰减函数
        decay_unload = lambda s: 1 - np.exp(-self.theta_2 * s)  # 卸货时的衰减函数

        # 货损成本计算公式：
        # C_31 表示运输与补电过程中的货损成本。
        # C_32 表示卸货过程中的货损成本。
        C_31 = self.p_5 * np.sum(self.customer_demand[:, None] * decay_transport(self.travel_time_matrix[:self.n, :self.n]))
        C_32 = self.p_5 * np.sum((self.customer_demand[:, None] - self.customer_demand[:, None]) * decay_unload(self.travel_time_matrix[:self.n, :self.n]))

        self.loss_cost = C_31 + C_32  # 总货损成本

        # 充电成本
        self.charging_cost = self.p_6 * np.sum(self.travel_time_matrix[:self.k_e])  # 电动车充电成本

        # 时间窗惩罚成本
        early_penalty = self.p_7 * np.sum(np.maximum(self.time_window_start[:, None] - self.travel_time_matrix[:self.n, :self.n], 0))  # 早到惩罚
        late_penalty = self.p_8 * np.sum(np.maximum(self.travel_time_matrix[:self.n, :self.n] - self.time_window_end[:, None], 0))  # 迟到惩罚
        self.time_window_penalty = early_penalty + late_penalty  # 总时间窗惩罚成本

        # 碳排放成本
        self.carbon_cost = self.c * np.sum(self.distance_matrix * self.E_ijk)  # 总碳排放成本

        # 总目标函数
        self.objective = self.fixed_cost + self.transport_cost + self.loss_cost + self.charging_cost + self.time_window_penalty + self.carbon_cost  # 所有成本的总和

        # 客户位置和配送中心位置
        self.customer_positions = np.array([[loc['x'], 'y'] for loc in locations if loc['type'] == 'c'])  # 客户位置数组
        depot_location = next(loc for loc in locations if loc['type'] == 'd')  # 配送中心位置
        self.depot_position = np.array([depot_location['x'], depot_location['y']])

        # 聚类初始化（例如，假设有两类）
        self.clusters = [[i for i in range(self.n // 2)], [i for i in range(self.n // 2, self.n)]]  # 将客户分成两类
        self.locations = locations
        self.vehicles = vehicles
        self.n_customers = len([loc for loc in locations if loc['type'] == 'c'])  # 客户数量
        self.location_id_to_index = {loc['id']: idx for idx, loc in enumerate(locations)}  # 位置 ID 到索引的映射
        self.distance_matrix = self.calculate_distance_matrix()  # 计算距离矩阵

    def calculate_distance_matrix(self):
        # 计算距离矩阵，假设使用欧氏距离
        n = len(self.locations)
        distance_matrix = np.zeros((n, n))  # 初始化距离矩阵
        for i, loc1 in enumerate(self.locations):
            for j, loc2 in enumerate(self.locations):
                distance_matrix[i, j] = np.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['y'] - loc2['y'])**2)  # 计算欧氏距离
        return distance_matrix

    def print_objective(self):
        # 打印目标函数值的方法（如果需要）
        pass
