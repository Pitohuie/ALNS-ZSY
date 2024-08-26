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
        return 0.0  # 电动车没有碳排放
    elif vehicle_type == 'fuel':
        return self.c * np.sum(self.d_ij[route[:-1], route[1:]] * self.F_ijk_f[route[:-1], route[1:]])
    else:
        raise ValueError("Unknown vehicle type.")
