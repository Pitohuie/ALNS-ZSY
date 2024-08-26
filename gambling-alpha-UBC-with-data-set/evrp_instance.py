import numpy as np


class CustomEVRPInstance:
    def __init__(self, locations):
        self.locations = locations

        # 创建一个位置 ID 到索引的映射
        self.location_id_to_index = {loc['id']: idx for idx, loc in enumerate(locations)}

        # 基础参数定义
        self.n = sum(1 for loc in locations if loc['type'] == 'c')  # 客户数量
        self.m = sum(1 for loc in locations if loc['type'] == 'f')  # 充电站数量
        self.O = next(loc for loc in locations if loc['type'] == 'd')  # 配送中心起点
        self.O_prime = self.n + self.m + 1  # 配送中心终点
        self.N = self.n + self.m + 2  # 所有节点数量，包括起点和终点

        # 参数（这些可以根据实际情况进行调整）
        self.Q_e = 100  # 电动车载重量
        self.Q_f = 150  # 燃油车载重量
        self.B_star = 200  # 电动车电池额定电量
        self.v_e = 60  # 电动车行驶速度
        self.v_f = 50  # 燃油车行驶速度
        self.m_v_e = 2000  # 电动车自重
        self.m_v_f = 2500  # 燃油车自重
        self.e = 0.5  # 电动车充电速度，单位时间充电量

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
        self.q_i = np.array([loc['demand'] for loc in locations if loc['type'] == 'c'])
        self.E_i = np.array([loc['ready_time'] for loc in locations if loc['type'] == 'c'])
        self.L_i = self.E_i + np.array([loc['due_date'] for loc in locations if loc['type'] == 'c'])

        # 距离矩阵和行驶时间矩阵
        coords = np.array([(loc['x'], loc['y']) for loc in locations])
        self.d_ij = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        self.t_ijk_e = self.d_ij / self.v_e
        self.t_ijk_f = self.d_ij / self.v_f

        # 电动车能耗计算
        c_d = 0.3  # 空气阻力系数
        rho = 1.225  # 空气密度 (kg/m^3)
        A = 2.5  # 车辆迎风面积 (m^2)
        g = 9.81  # 重力加速度 (m/s^2)
        phi_d = 0.9  # 电池放电效率
        varphi_d = 0.85  # 电机效率

        K_ijk_e = 0.5 * c_d * rho * A * self.v_e ** 3 + (self.m_v_e + self.q_i.mean()) * g * c_d * self.v_e
        self.L_ijk_e = phi_d * varphi_d * K_ijk_e * self.t_ijk_e

        # 燃油车油耗计算
        xi = 14.7  # 燃油空气质量比
        kappa = 44.8  # 燃油热值 (kJ/g)
        psi = 0.85  # 燃油转换系数
        sigma = 0.5  # 发动机摩擦系数 (kJ/r/L)
        vartheta = 30  # 发动机转速 (r/s)
        omega = 2.0  # 发动机排量 (L)
        eta = 0.3  # 燃油机效率
        tau = 0.9  # 车传动系统效率

        K_ijk_f = 0.5 * c_d * rho * A * self.v_f ** 3 + (self.m_v_f + self.q_i.mean()) * g * c_d * self.v_f
        self.F_ijk_f = (xi / (kappa * psi)) * (sigma * vartheta * omega + K_ijk_f / (eta * tau))

        # 碳排放量计算
        pi_e = 2.31  # 燃油排放系数 (kg CO2/L)
        self.E_ijk_f = pi_e * self.F_ijk_f

        # 初始化决策变量
        self.initialize_decision_variables()

    def initialize_decision_variables(self):
        # 初始化决策变量
        self.K_e = 0  # 初始电动车数量
        self.K_f = 0  # 初始燃油车数量
        self.x_ijk = np.zeros((self.N, self.N, self.K_e + self.K_f), dtype=int)  # 路径决策变量，0或1
        self.u_ijk = np.zeros((self.N, self.N, self.K_e + self.K_f))  # 载重量变量，≥0
        self.f_ijk = np.zeros((self.N, self.N, self.K_e))  # 电动车充电电量变量，≥0
        self.B_ik1 = np.ones((self.N, self.K_e)) * self.B_star  # 初始化电量状态为满电量
        self.B_ik2 = np.ones((self.N, self.K_e)) * self.B_star  # 初始化电量状态为满电量
        self.a_ik = np.zeros((self.N, self.K_e + self.K_f))  # 车辆到达节点时间
        self.b_ik = np.zeros((self.N, self.K_e + self.K_f))  # 车辆离开节点时间
        self.w_ik = np.zeros((self.N, self.K_e + self.K_f))  # 车辆在节点等待时间，≥0
        self.T_ik = np.zeros((self.N, self.K_e))  # 车辆在充电桩充电时间

    def calculate_fixed_cost(self):
        return self.p_1 * self.K_e + self.p_2 * self.K_f

    def calculate_transport_cost(self):
        transport_cost_e = self.p_3 * np.sum(self.d_ij)  # 电动车运输成本
        transport_cost_f = self.p_4 * np.sum(self.d_ij)  # 燃油车运输成本
        return transport_cost_e + transport_cost_f

    def calculate_loss_cost(self):
        decay_transport = lambda t: 1 - np.exp(-self.theta_1 * t)  # 运输与补电时的衰减函数
        decay_unload = lambda s: 1 - np.exp(-self.theta_2 * s)  # 卸货时的衰减函数

        C_31 = self.p_5 * np.sum(self.q_i[:, None] * decay_transport(self.t_ijk_e[:self.n, :self.n]))
        C_32 = self.p_5 * np.sum((self.q_i[:, None] - self.q_i[:, None]) * decay_unload(self.t_ijk_e[:self.n, :self.n]))
        return C_31 + C_32

    def calculate_charging_cost(self):
        # 计算充电电量（推导出充电电量）
        f_ijk = np.maximum(0, self.B_star - self.B_ik1)
        return self.p_6 * np.sum(f_ijk)

    def calculate_time_window_penalty(self):
        early_penalty = self.p_7 * np.sum(np.maximum(self.E_i[:, None] - self.t_ijk_e[:self.n, :self.n], 0))
        late_penalty = self.p_8 * np.sum(np.maximum(self.t_ijk_e[:self.n, :self.n] - self.L_i[:, None], 0))
        return early_penalty + late_penalty

    def calculate_carbon_cost(self):
        return self.c * np.sum(self.d_ij * self.E_ijk_f)

    def calculate_total_cost(self):
        fixed_cost = self.calculate_fixed_cost()
        transport_cost = self.calculate_transport_cost()
        loss_cost = self.calculate_loss_cost()
        charging_cost = self.calculate_charging_cost()
        time_window_penalty = self.calculate_time_window_penalty()
        carbon_cost = self.calculate_carbon_cost()
        return fixed_cost + transport_cost + loss_cost + charging_cost + time_window_penalty + carbon_cost

    def print_objective(self):
        total_cost = self.calculate_total_cost()
        print("Objective Function Value: ", total_cost)


# 示例数据
locations = [
    {'id': 1, 'type': 'd', 'x': 0, 'y': 0, 'demand': 0, 'ready_time': 0, 'due_date': 0},
    {'id': 2, 'type': 'c', 'x': 1, 'y': 1, 'demand': 10, 'ready_time': 0, 'due_date': 10},
    {'id': 3, 'type': 'c', 'x': 2, 'y': 2, 'demand': 20, 'ready_time': 0, 'due_date': 10},
    {'id': 4, 'type': 'f', 'x': 3, 'y': 3, 'demand': 0, 'ready_time': 0, 'due_date': 0},
]

# 创建实例
instance = CustomEVRPInstance(locations)

# 打印目标函数值
instance.print_objective()
