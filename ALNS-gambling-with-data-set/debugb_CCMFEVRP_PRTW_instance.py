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
        # 初始化customer_positions和depot_position
        self.customer_positions = np.array([(loc.x, loc.y) for loc in self.locations if loc.type == 'c'])
        depot = next(loc for loc in self.locations if loc.type == 'd')
        self.depot_position = np.array([depot.x, depot.y])

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

        pi_e = 2.31
        self.E_ijk_f = pi_e * self.F_ijk_f

        self.initialize_decision_variables()

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

    def calculate_fixed_cost(self, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return self.p_1 * self.K_e
        elif vehicle_type == 'fuel':
            return self.p_2 * self.K_f
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_transport_cost(self, route, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return self.p_3 * np.sum(self.d_ij[route[:-1], route[1:]])
        elif vehicle_type == 'fuel':
            return self.p_4 * np.sum(self.d_ij[route[:-1], route[1:]])
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_loss_cost(self, route, vehicle_type: str) -> float:
        decay_transport = lambda t: 1 - np.exp(-self.theta_1 * t)
        decay_unload = lambda s: 1 - np.exp(-self.theta_2 * s)

        if vehicle_type == 'electric':
            C_31 = self.p_5 * np.sum(self.q_i[:, None] * decay_transport(self.t_ijk_e[route[:-1], route[1:]]))
            C_32 = self.p_5 * np.sum((self.q_i[:, None] - self.q_i[:, None]) * decay_unload(self.t_ijk_e[route[:-1], route[1:]]))
        elif vehicle_type == 'fuel':
            C_31 = self.p_5 * np.sum(self.q_i[:, None] * decay_transport(self.t_ijk_f[route[:-1], route[1:]]))
            C_32 = self.p_5 * np.sum((self.q_i[:, None] - self.q_i[:, None]) * decay_unload(self.t_ijk_f[route[:-1], route[1:]]))
        else:
            raise ValueError("Unknown vehicle type.")
        return C_31 + C_32

    def calculate_charging_cost(self, route, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            f_ijk = np.maximum(0, self.B_star - self.B_ik1)
            return self.p_6 * np.sum(f_ijk)
        else:
            return 0.0  # 燃油车没有充电成本

    def calculate_time_window_penalty(self, route, vehicle_type: str) -> float:
        early_penalty = self.p_7 * np.sum(np.maximum(self.E_i[:, None] - self.t_ijk_e[route[:-1], route[1:]], 0))
        late_penalty = self.p_8 * np.sum(np.maximum(self.t_ijk_e[route[:-1], route[1:]] - self.L_i[:, None], 0))
        return early_penalty + late_penalty

    def calculate_carbon_cost(self, route, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return self.c * np.sum(self.d_ij[route[:-1], route[1:]] * self.E_ijk_f[route[:-1], route[1:]])
        elif vehicle_type == 'fuel':
            return self.c * np.sum(self.d_ij[route[:-1], route[1:]] * self.F_ijk_f[route[:-1], route[1:]])
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_total_cost(self, route, vehicle_type: str) -> float:
        fixed_cost = self.calculate_fixed_cost(vehicle_type)
        transport_cost = self.calculate_transport_cost(route, vehicle_type)
        loss_cost = self.calculate_loss_cost(route, vehicle_type)
        charging_cost = self.calculate_charging_cost(route, vehicle_type)
        time_window_penalty = self.calculate_time_window_penalty(route, vehicle_type)
        carbon_cost = self.calculate_carbon_cost(route, vehicle_type)

        return fixed_cost + transport_cost + loss_cost + charging_cost + time_window_penalty + carbon_cost

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
