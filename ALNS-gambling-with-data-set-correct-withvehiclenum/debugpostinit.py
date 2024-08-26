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

    self.initialize_decision_variables()
