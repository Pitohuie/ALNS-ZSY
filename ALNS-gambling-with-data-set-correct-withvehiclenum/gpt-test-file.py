import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# 定义Location类
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

# 定义CustomEVRPInstance类
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
    K_e: int = 1
    K_f: int = 1
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
    minimum_battery_threshold: float = 20

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
        self.precomputed_f_ijk = np.maximum(0, self.B_star - self.B_ik1)

        self.customer_positions = np.array([(loc.x, loc.y) for loc in self.locations if loc.type == 'c'])
        depot = next(loc for loc in self.locations if loc.type == 'd')
        self.depot_position = np.array([depot.x, depot.y])

        self.calculate_energy_consumption_factors()

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

    def calculate_energy_consumption_factors(self):
        c_d = 0.3
        rho = 1.225
        A = 2.5
        g = 9.81
        phi_d = 0.9
        varphi_d = 0.85

        K_ijk_e = 0.5 * c_d * rho * A * self.v_e ** 3 + (self.m_v_e + self.q_i.mean()) * g * c_d * self.v_e
        self.L_ijk_e = phi_d * varphi_d * K_ijk_e * self.t_ijk_e

        xi = 14.7
        kappa = 44.8
        psi = 0.85
        sigma = 0.5
        vartheta = 30
        omega = 2.0
        eta = 0.3
        tau = 0.9

        K_ijk_f = 0.5 * c_d * rho * A * self.v_f ** 3 + (self.m_v_f + self.q_i.mean()) * g * c_d * self.v_f
        self.F_ijk_f = (xi / (kappa * psi)) * (sigma * vartheta * omega + K_ijk_f / (eta * tau))
        self.F_ijk_f = np.full_like(self.d_ij, self.F_ijk_f)

        pi_e = 2.31
        self.E_ijk_f = pi_e * self.F_ijk_f

    def calculate_battery_usage(self, route):
        battery_usage = 0.0
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            battery_usage += self.L_ijk_e[start, end]
        return battery_usage

# 读取Solomon实例的函数
def read_solomon_instance(file_path):
    locations = []
    vehicles = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == 'StringID':
                continue
            if line.startswith('Q Vehicle fuel tank capacity'):
                vehicles['fuel_tank_capacity'] = float(parts[-1].strip('/'))
            elif line.startswith('C Vehicle load capacity'):
                vehicles['load_capacity'] = float(parts[-1].strip('/'))
            elif line.startswith('r fuel consumption rate'):
                vehicles['fuel_consumption_rate'] = float(parts[-1].strip('/'))
            elif line.startswith('g inverse refueling rate'):
                vehicles['inverse_refueling_rate'] = float(parts[-1].strip('/'))
            elif line.startswith('v average Velocity'):
                vehicles['average_velocity'] = float(parts[-1].strip('/'))
            else:
                if len(parts) == 8:
                    location = {
                        'id': parts[0],
                        'type': parts[1],
                        'x': float(parts[2]),
                        'y': float(parts[3]),
                        'demand': float(parts[4]),
                        'ready_time': float(parts[5]),
                        'due_date': float(parts[6]),
                        'service_time': float(parts[7])
                    }
                    locations.append(location)

    return locations, vehicles

# 调试版本的贪婪算法
def basic_greedy_initial_solution_with_depot(instance, vehicle_type, customers):
    routes = []
    unassigned_customers = set(customers)
    remaining_battery = instance.B_star
    iteration_count = 0

    depot_index = instance.location_id_to_index[instance.O.id]

    while unassigned_customers:
        current_route = [depot_index]  # Start from the depot
        remaining_capacity = instance.Q_e if vehicle_type == 'electric' else instance.Q_f

        while unassigned_customers:
            iteration_count += 1
            best_customer = None
            best_cost = float('inf')

            for customer in unassigned_customers:
                if instance.q_i[customer - 1] <= remaining_capacity:
                    last_node = current_route[-1]
                    distance_cost = instance.d_ij[last_node, customer]

                    if distance_cost < best_cost:
                        best_cost = distance_cost
                        best_customer = customer

            if best_customer is None:
                break

            current_route.append(best_customer)
            unassigned_customers.remove(best_customer)
            remaining_capacity -= instance.q_i[best_customer - 1]

        current_route.append(depot_index)  # End at the depot
        routes.append(current_route)
        print(f"Iteration {iteration_count}: Generated route {current_route} with cost {best_cost}")

    return routes

# 执行初始解生成
def construct_basic_initial_solution_with_depot(instance: CustomEVRPInstance):
    electric_customers = list(range(1, 6))  # 测试使用的客户数量
    fuel_customers = list(range(1, 6))

    print("Generating initial solutions for electric vehicles with depot...")
    electric_routes = basic_greedy_initial_solution_with_depot(instance, 'electric', electric_customers)
    print("Generating initial solutions for fuel vehicles with depot...")
    fuel_routes = basic_greedy_initial_solution_with_depot(instance, 'fuel', fuel_customers)

    return electric_routes, fuel_routes

# 加载实例数据并生成初始解
file_path = "c101_21.txt"  # 实际路径可能需要调整
locations_data, vehicles_data = read_solomon_instance(file_path)

locations = [Location(**loc) for loc in locations_data]
instance = CustomEVRPInstance(locations, vehicles_data)

electric_routes, fuel_routes = construct_basic_initial_solution_with_depot(instance)

print(f"Electric Routes: {electric_routes}")
print(f"Fuel Routes: {fuel_routes}")
