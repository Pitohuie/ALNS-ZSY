from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from a_read_instance import read_solomon_instance
import time

@dataclass
class Location:
    id: int
    type: str
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float

@dataclass
class CustomEVRPInstance:
    locations: List[Location]
    vehicles: Dict[str, float]
    location_id_to_index: Dict[int, int] = field(init=False)
    n: int = field(init=False)
    m: int = field(init=False)
    O: Location = field(init=False)
    O_prime: int = field(init=False)
    N: int = field(init=False)
    Q_e: float = field(init=False)
    Q_f: float = field(init=False)
    B_star: float = 200
    v_e: float = field(init=False)
    v_f: float = field(init=False)
    m_v_e: float = 2000
    m_v_f: float = 2500
    e: float = field(init=False)
    p_1: float = 500
    p_2: float = 400
    p_3: float = 2
    p_4: float = 3
    p_5: float = 10
    theta_1: float = 0.01
    theta_2: float = 0.02
    c: float = 0.5
    p_6: float = 0.1
    p_7: float = 50
    p_8: float = 50
    soc_min: float = 0.2
    M: float = 1e6
    q_i: np.ndarray = field(init=False)
    E_i: np.ndarray = field(init=False)
    L_i: np.ndarray = field(init=False)
    d_ij: np.ndarray = field(init=False)
    t_ijk_e: np.ndarray = field(init=False)
    t_ijk_f: np.ndarray = field(init=False)
    L_ijk_e: np.ndarray = field(init=False)
    F_ijk_f: np.ndarray = field(init=False)
    E_ijk_f: np.ndarray = field(init=False)
    K_e: int = 0
    K_f: int = 0
    x_ijk: np.ndarray = field(init=False)
    u_ijk: np.ndarray = field(init=False)
    f_ijk: np.ndarray = field(init=False)
    B_ik1: np.ndarray = field(init=False)
    B_ik2: np.ndarray = field(init=False)
    a_ik: np.ndarray = field(init=False)
    b_ik: np.ndarray = field(init=False)
    w_ik: np.ndarray = field(init=False)
    T_ik: np.ndarray = field(init=False)
    customer_positions: np.ndarray = field(init=False)
    depot_position: np.ndarray = field(init=False)
    precomputed_f_ijk: np.ndarray = field(init=False)
    minimum_battery_threshold: float = 20  # 假设的值，可以根据实际情况调整

    def __post_init__(self):
        self.location_id_to_index = {loc.id: idx for idx, loc in enumerate(self.locations)}
        self.n = sum(1 for loc in self.locations if loc.type == 'c')
        self.m = sum(1 for loc in self.locations if loc.type == 'f')
        self.O = next(loc for loc in self.locations if loc.type == 'd')
        self.O_prime = self.n + self.m + 1
        self.N = self.n + self.m + 2

        self.Q_e = self.vehicles['load_capacity']
        self.Q_f = self.vehicles['load_capacity']
        self.v_e = self.vehicles['average_velocity']
        self.v_f = self.vehicles['average_velocity']
        self.e = self.vehicles['inverse_refueling_rate']

        self.q_i = np.array([loc.demand for loc in self.locations if loc.type == 'c'])
        self.E_i = np.array([loc.ready_time for loc in self.locations if loc.type == 'c'])
        self.L_i = self.E_i + np.array([loc.due_date for loc in self.locations if loc.type == 'c'])

        coords = np.array([(loc.x, loc.y) for loc in self.locations])
        self.d_ij = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        self.t_ijk_e = self.d_ij / self.v_e
        self.t_ijk_f = self.d_ij / self.v_f
        self.initialize_decision_variables()
        # 预计算 f_ijk
        self.precomputed_f_ijk = np.maximum(0, self.B_star - self.B_ik1)

        # 初始化 customer_positions 和 depot_position
        self.customer_positions = np.array([(loc.x, loc.y) for loc in self.locations if loc.type == 'c'])
        depot = next(loc for loc in self.locations if loc.type == 'd')
        self.depot_position = np.array([depot.x, depot.y])

        # 公共参数
        c_d = 0.3  # 空气阻力系数
        rho = 1.225  # 空气密度 (kg/m^3)
        A = 2.5  # 车辆迎风面积 (m^2)
        g = 9.81  # 重力加速度 (m/s^2)
        phi_d = 0.9  # 空气动力学效率系数
        varphi_d = 0.85  # 车辆动力系统效率系数

        ### 电动车的能量消耗计算 ###
        K_ijk_e = 0.5 * c_d * rho * A * self.v_e ** 3 + (self.m_v_e + self.q_i.mean()) * g * c_d * self.v_e
        self.L_ijk_e = phi_d * varphi_d * K_ijk_e * self.t_ijk_e

        ### 燃油车的能量消耗计算 ###
        xi = 14.7  # 燃油热值相关系数
        kappa = 44.8  # 燃油效率相关系数
        psi = 0.85  # 发动机效率相关系数
        sigma = 0.5  # 燃油效率系数
        vartheta = 30  # 发动机功率系数
        omega = 2.0  # 发动机转速系数
        eta = 0.3  # 传动系统效率
        tau = 0.9  # 总体效率系数

        K_ijk_f = 0.5 * c_d * rho * A * self.v_f ** 3 + (self.m_v_f + self.q_i.mean()) * g * c_d * self.v_f
        self.F_ijk_f = (xi / (kappa * psi)) * (sigma * vartheta * omega + K_ijk_f / (eta * tau))
        self.F_ijk_f = np.full_like(self.d_ij, self.F_ijk_f)

        # 电动车没有碳排放，燃油车则根据 F_ijk_f 计算
        pi_e = 2.31  # 电动车的单位能量价格
        self.E_ijk_f = pi_e * self.F_ijk_f



    def initialize_decision_variables(self):
        self.x_ijk = np.zeros((self.N, self.N, self.K_e + self.K_f), dtype=int)
        self.u_ijk = np.zeros((self.N, self.N, self.K_e + self.K_f))
        self.f_ijk = np.zeros((self.N, self.N, self.K_e))
        self.B_ik1 = np.ones((self.N, self.K_e)) * self.B_star
        self.B_ik2 = np.ones((self.N, self.K_e)) * self.B_star
        self.a_ik = np.zeros((self.N, self.K_e + self.K_f))
        self.b_ik = np.zeros((self.N, self.K_e + self.K_f))
        self.w_ik = np.zeros((self.N, self.K_e + self.K_f))
        self.T_ik = np.zeros((self.N, self.K_e))

    def calculate_energy_consumption(self, route):
        total_energy = 0
        for i in range(len(route) - 1):
            total_energy += self.L_ijk_e[route[i], route[i + 1]]
        return total_energy

    def calculate_battery_usage(self, route):
        """
        根据路径计算电池的使用量。
        假设电池使用量与路径上的能耗成正比。
        """
        battery_usage = 0.0
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            # 使用你在 b_CCMFEVRP_PRTW_instance 中定义的 L_ijk_e 或其他变量来计算电池消耗
            battery_usage += self.L_ijk_e[start, end]

        return battery_usage
    def calculate_fuel_consumption(self, route):
        total_fuel = 0
        for i in range(len(route) - 1):
            total_fuel += self.F_ijk_f[route[i], route[i + 1]]
        return total_fuel

    def calculate_total_cost(self, route, vehicle_type: str) -> float:
        fixed_cost = self.calculate_fixed_cost(vehicle_type)
        transport_cost = self.calculate_transport_cost(route, vehicle_type)
        loss_cost = self.calculate_loss_cost(route, vehicle_type)
        charging_cost = self.calculate_charging_cost(route, vehicle_type)
        time_window_penalty = self.calculate_time_window_penalty(route, vehicle_type)
        carbon_cost = self.calculate_carbon_cost(route, vehicle_type)

        return fixed_cost + transport_cost + loss_cost + charging_cost + time_window_penalty + carbon_cost

    def calculate_fixed_cost(self, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return self.p_1 * self.K_e
        elif vehicle_type == 'fuel':
            return self.p_2 * self.K_f
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_transport_cost(self, route, vehicle_type: str) -> float:
        if len(route) < 2:
            return 0.0  # 防止路径过短

        print(f"Route: {route}")
        print(f"Indexed d_ij: {self.d_ij[route[:-1], route[1:]]}")

        if vehicle_type == 'electric':
            return self.p_3 * np.sum(self.d_ij[route[:-1], route[1:]])
        elif vehicle_type == 'fuel':
            return self.p_4 * np.sum(self.d_ij[route[:-1], route[1:]])
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_loss_cost(self, route, vehicle_type: str) -> float:
        total_loss_cost = 0.0
        chunk_size = 10

        for i in range(0, len(route) - 1, chunk_size):
            route_chunk = route[i:i + chunk_size + 1]

            if vehicle_type == 'electric':
                C_31 = self.p_5 * np.sum(
                    self.q_i[:, None] * self.decay_transport(self.t_ijk_e[route_chunk[:-1], route_chunk[1:]]))
                C_32 = self.p_5 * np.sum(
                    (self.q_i[:, None] - self.q_i[:, None]) * self.decay_unload(
                        self.t_ijk_e[route_chunk[:-1], route_chunk[1:]]))
            elif vehicle_type == 'fuel':
                C_31 = self.p_5 * np.sum(
                    self.q_i[:, None] * self.decay_transport(self.t_ijk_f[route_chunk[:-1], route_chunk[1:]]))
                C_32 = self.p_5 * np.sum(
                    (self.q_i[:, None] - self.q_i[:, None]) * self.decay_unload(
                        self.t_ijk_f[route_chunk[:-1], route_chunk[1:]]))
            else:
                raise ValueError("Unknown vehicle type.")

            total_loss_cost += C_31 + C_32

        return total_loss_cost

    def calculate_charging_cost(self, route, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            total_cost = 0.0

            # 设置一个较小的块大小
            chunk_size = 5

            # 仅处理路径中涉及的部分
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                for j in range(0, self.B_ik1.shape[1], chunk_size):
                    f_ijk_chunk = np.maximum(0, self.B_star - self.B_ik1[current_node, j:j + chunk_size])
                    total_cost += np.sum(f_ijk_chunk)

            return self.p_6 * total_cost
        else:
            return 0.0  # 燃油车没有充电成本

    def decay_transport(self, transport_time, decay_rate=0.01):
        """
        模拟运输过程中货物的损耗衰减。

        transport_time: 运输时间
        decay_rate: 衰减率，默认值为0.01，表示每单位时间的损耗率
        """
        return np.exp(-decay_rate * transport_time)  # 指数衰减

    def decay_unload(self, unload_time, decay_rate=0.02):
        """
        模拟装卸过程中货物的损耗衰减。

        unload_time: 装卸时间
        decay_rate: 衰减率，默认值为0.02，表示每单位时间的损耗率
        """
        return np.exp(-decay_rate * unload_time)  # 指数衰减

    # 之后的代码...

    def calculate_time_window_penalty(self, route, vehicle_type: str) -> float:
        if len(route) < 2:
            return 0.0

        t_diff_e = self.t_ijk_e[route[:-1], route[1:]]
        t_diff_f = self.t_ijk_f[route[:-1], route[1:]]

        print(f"t_diff_e shape: {t_diff_e.shape}")
        print(f"t_diff_f shape: {t_diff_f.shape}")
        print(f"E_i shape: {self.E_i[:, None].shape}")

        if vehicle_type == 'electric':
            early_penalty = self.p_7 * np.sum(np.maximum(self.E_i[:, None] - t_diff_e, 0))
            late_penalty = self.p_8 * np.sum(np.maximum(t_diff_e - self.L_i[:, None], 0))
        elif vehicle_type == 'fuel':
            early_penalty = self.p_7 * np.sum(np.maximum(self.E_i[:, None] - t_diff_f, 0))
            late_penalty = self.p_8 * np.sum(np.maximum(t_diff_f - self.L_i[:, None], 0))
        else:
            raise ValueError("Unknown vehicle type.")

        return early_penalty + late_penalty

    def calculate_carbon_cost(self, route, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return 0.0  # 电动车没有碳排放
        elif vehicle_type == 'fuel':
            driving_segments = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            carbon_emissions = self.c * np.sum(
                [self.d_ij[start, end] * self.F_ijk_f[start, end] for start, end in driving_segments])
            return carbon_emissions
        else:
            raise ValueError("Unknown vehicle type.")

    def print_objective(self, route, vehicle_type: str):
        total_cost = self.calculate_total_cost(route, vehicle_type)
        print(f"Objective Function Value for {vehicle_type}: ", total_cost)

if __name__ == "__main__":
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)
    route = [0, 1, 2, 3, 4, 5]  # 示例路径
    instance.print_objective(route, 'electric')
    instance.print_objective(route, 'fuel')
