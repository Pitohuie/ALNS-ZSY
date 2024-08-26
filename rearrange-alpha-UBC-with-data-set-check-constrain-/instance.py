import numpy as np


class CustomEVRPInstance:
    def __init__(self, locations, vehicles, sigma=1):
        self.locations = locations.copy()
        self.vehicles = vehicles
        self.sigma = sigma

        self.location_id_to_index = {loc['id']: idx for idx, loc in enumerate(locations)}

        self.n = sum(1 for loc in locations if loc['type'] == 'c')  # 客户数量
        self.m = sum(1 for loc in locations if loc['type'] == 'f')  # 充电站数量
        self.O = next(i for i, loc in enumerate(locations) if loc['type'] == 'd')  # 配送中心起点
        self.O_prime = len(locations)
        self.locations.append(locations[self.O].copy())  # 添加配送中心终点

        for i in range(self.m):
            original_station = locations[self.n + i]
            for _ in range(sigma):
                copy_station = original_station.copy()
                copy_station['id'] = f"{copy_station['id']}_copy_{_}"
                self.locations.append(copy_station)

        self.N = self.n + self.m * (sigma + 1) + 1  # 所有节点数量，包括起点和终点

        self.Q_e = vehicles['fuel_tank_capacity']  # 电动车载重量
        self.Q_f = vehicles['load_capacity']  # 燃油车载重量
        self.B_star = 200  # 电动车电池额定电量
        self.v = vehicles['average_velocity']  # 车辆行驶速度
        self.m_v = 2000  # 车辆自重
        self.e = vehicles['inverse_refueling_rate']  # 车辆充电速度

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

        self.customer_demand = np.array([loc['demand'] for loc in locations if loc['type'] == 'c'])
        self.time_window_start = np.array([loc['ready_time'] for loc in locations if loc['type'] == 'c'])
        self.time_window_end = np.array([loc['due_date'] for loc in locations if loc['type'] == 'c'])
        self.service_time = np.array([loc['service_time'] for loc in locations if loc['type'] == 'c'])
        self.service_time = np.concatenate((self.service_time, np.zeros(self.N - len(self.service_time))))

        coords = np.array([(loc['x'], loc['y']) for loc in self.locations])
        self.distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        self.travel_time_matrix = self.distance_matrix / self.v

        self.x = None
        self.u = None
        self.a = None
        self.b = None
        self.B = None
        self.f = None

        c_d = 0.3
        rho = 1.225
        A = 2.5
        g = 9.81
        phi_d = 0.9
        varphi_d = 0.85

        K_ijk = 0.5 * c_d * rho * A * self.v ** 3 + (self.m_v + self.customer_demand.mean()) * g * c_d * self.v
        self.L_ijk = phi_d * varphi_d * K_ijk * self.travel_time_matrix

        xi = 14.7
        kappa = 44.8
        psi = 0.85
        sigma = 0.5
        vartheta = 30
        omega = 2.0
        eta = 0.3
        tau = 0.9

        self.F_ijk = (xi / (kappa * psi)) * (sigma * vartheta * omega + K_ijk / (eta * tau))

        pi_e = 2.31
        self.E_ijk = pi_e * self.F_ijk


    def initialize_variables(self, k_e, k_f):
        self.k_e = k_e
        self.k_f = k_f
        self.x = np.zeros((self.N, self.N, self.k_e + self.k_f), dtype=int)
        self.u = np.zeros((self.N, self.N, self.k_e + self.k_f), dtype=float)
        self.a = np.zeros((self.N, self.k_e + self.k_f), dtype=float)
        self.b = np.zeros((self.N, self.k_e + self.k_f), dtype=float)
        self.B = np.zeros((self.N, self.k_e), dtype=float)
        self.f = np.zeros((self.N, self.N, self.k_e), dtype=float)


 def calculate_fixed_cost(self):
        K_e = list(range(self.k_e))
        K_f = list(range(self.k_e, self.k_e + self.k_f))
        C1 = self.p_1 * sum(sum(self.x[0, j, k] for j in range(self.N)) for k in K_e) + \
             self.p_2 * sum(sum(self.x[0, j, k] for j in range(self.N)) for k in K_f)
        return C1

    def calculate_transport_cost(self):
        C2 = self.p_3 * sum(self.distance_matrix[i][j] * self.x[i, j, k]
                            for k in range(self.k_e)
                            for i in range(self.N)
                            for j in range(self.N) if i != j) + \
             self.p_4 * sum(self.distance_matrix[i][j] * self.x[i, j, k]
                            for k in range(self.k_e, self.k_e + self.k_f)
                            for i in range(self.N)
                            for j in range(self.N) if i != j)
        return C2

    def calculate_loss_cost(self):
        decay_transport = lambda t: 1 - np.exp(-self.theta_1 * t)
        decay_unload = lambda s: 1 - np.exp(-self.theta_2 * s)

        C3_1 = self.p_5 * sum(self.x[i, j, k] * self.customer_demand[j] *
                              decay_transport(self.a[j, k] - self.b[0, k] + self.service_time[j])
                              for i in range(self.N)
                              for j in range(self.N) if i != j
                              for k in range(self.k_e + self.k_f))

        C3_2 = self.p_5 * sum(self.x[i, j, k] * (self.u[i, j, k] - self.customer_demand[j]) *
                              decay_unload(self.service_time[j])
                              for i in range(self.N)
                              for j in range(self.N) if i != j
                              for k in range(self.k_e + self.k_f))

        C3 = C3_1 + C3_2
        return C3

    def calculate_charging_cost(self):
        C4 = self.p_6 * sum(self.x[i, j, k] * self.f[i, j, k]
                            for i in range(self.N)
                            for j in range(self.N) if i != j
                            for k in range(self.k_e))  # 仅电动车有充电成本
        return C4

    def calculate_time_window_penalty_cost(self):
        C5 = sum(self.p_7 * max(self.time_window_start[i] - self.a[i, k], 0) +
                 self.p_8 * max(self.a[i, k] - self.time_window_end[i], 0)
                 for i in range(self.n)
                 for k in range(self.k_e + self.k_f))
        return C5

    def calculate_carbon_cost(self):
        C6 = self.c * sum(self.x[i, j, k] * self.E_ijk[i, j]
                          for i in range(self.N)
                          for j in range(self.N) if i != j
                          for k in range(self.k_e, self.k_e + self.k_f) if k >= self.k_e)  # 仅传统车辆有碳排放成本
        return C6

    def calculate_total_cost(self):
        C1 = self.calculate_fixed_cost()
        C2 = self.calculate_transport_cost()
        C3 = self.calculate_loss_cost()
        C4 = self.calculate_charging_cost()
        C5 = self.calculate_time_window_penalty_cost()
        C6 = self.calculate_carbon_cost()
        total_cost = C1 + C2 + C3 + C4 + C5 + C6
        return total_cost
def node_visit_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    M = instance.M

    constraints.append(
        sum(instance.x[0, j, k] for j in C + R for k in K) == sum(instance.x[j, instance.O_prime, k] for j in C + R for k in K)
    )

    for i in C:
        constraints.append(
            sum(instance.x[i, j, k] for j in N if j != i for k in K) == 1
        )

    for j in C:
        for k in K:
            constraints.append(
                sum(instance.x[i, j, k] for i in N if i != j) == sum(instance.x[j, i, k] for i in N if i != j)
            )

    for r in R:
        for k in K_e:
            constraints.append(
                sum(instance.x[i, r, k] for i in N if i != r) <= 1
            )
            constraints.append(
                sum(instance.x[r, j, k] for j in N if j != r) <= 1
            )

    return constraints

def load_balance_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    E = [(i, j) for i in N for j in N if i != j]
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    K_f = list(range(instance.k_e, instance.k_e + instance.k_f))
    M = instance.M
    Q_e = instance.Q_e
    Q_f = instance.Q_f
    q = instance.customer_demand

    for (i, j) in E:
        for k in K_e:
            constraints.append(instance.u[i, j, k] <= Q_e)
        for k in K_f:
            constraints.append(instance.u[i, j, k] <= Q_f)

    for j in C:
        for k in K:
            constraints.append(
                sum(instance.u[i, j, k] for i in N if i != j) - sum(instance.u[j, i, k] for i in N if i != j) + M * (1 - sum(instance.x[i, j, k] for i in N if i != j)) >= q[j]
            )

    for k in K:
        constraints.append(
            sum(instance.u[i, instance.O_prime, k] for i in C + R) == 0
        )

    return constraints

def time_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    M = instance.M
    t = instance.travel_time_matrix
    s = instance.service_time
    w = np.zeros_like(s)
    E = [loc['ready_time'] for loc in instance.locations]
    L = [loc['due_date'] for loc in instance.locations]

    for k in K:
        constraints.append(instance.b[0, k] >= E[0])

    for j in C:
        for k in K:
            constraints.append(
                instance.b[0, k] + t[0, j] * instance.x[0, j, k] - M * (1 - instance.x[0, j, k]) <= instance.a[j, k]
            )

    for i in C:
        for j in N:
            if j != i:
                for k in K:
                    constraints.append(
                        instance.a[i, k] + t[i, j] * instance.x[i, j, k] + s[i] + w[i, k] - M * (1 - instance.x[i, j, k]) <= instance.a[j, k]
                    )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.b[r, k] >= instance.a[r, k] + instance.B_star / instance.e
            )

    for i in N:
        for k in K:
            constraints.append(
                instance.b[i, k] + t[i, instance.O_prime] * instance.x[i, instance.O_prime, k] - M * (1 - instance.x[i, instance.O_prime, k]) <= L[0]
            )

    for i in C:
        for k in K:
            constraints.append(
                instance.b[i, k] == instance.a[i, k] + s[i] + w[i, k]
            )

    for i in R:
        for k in K_e:
            constraints.append(
                instance.b[i, k] == instance.a[i, k] + M
            )

    for i in C:
        for k in K_e:
            constraints.append(
                E[i] <= instance.a[i, k] <= L[i]
            )

    return constraints

def battery_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K_e = list(range(instance.k_e))
    M = instance.M
    L_ijk = instance.L_ijk
    B_star = instance.B_star
    soc_min = instance.soc_min

    for i in N:
        for j in N:
            if j != i:
                for k in K_e:
                    constraints.append(
                        instance.B[i, k] - L_ijk[i, j] * instance.x[i, j, k] + B_star * (1 - instance.x[i, j, k]) >= instance.B[j, k]
                    )

    for k in K_e:
        constraints.append(
            instance.B[0, k] == B_star
        )

    for i in N:
        for k in K_e:
            constraints.append(
                instance.B[i, k] >= B_star * soc_min
            )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.B[r, k] == B_star
            )

    return constraints

def variable_constraints(instance):
    constraints = []

    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))

    for (i, j) in [(i, j) for i in N for j in N if i != j]:
        for k in K:
            constraints.append(instance.u[i, j, k] >= 0)
            if k in range(instance.k_e):
                constraints.append(instance.f[i, j, k] >= 0)
            constraints.append(instance.x[i, j, k] == 0 or instance.x[i, j, k] == 1)

    return constraints

def additional_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    M = instance.M

    for k in K:
        constraints.append(instance.b[0, k] >= instance.time_window_start[0])

    for j in C:
        for k in K:
            constraints.append(
                instance.b[0, k] + instance.travel_time_matrix[0, j] * instance.x[0, j, k] <= instance.a[j, k]
            )

    for i in C:
        for j in N:
            if j != i:
                for k in K:
                    constraints.append(
                        instance.a[i, k] + instance.service_time[i] + instance.travel_time_matrix[i, j] * instance.x[i, j, k] <= instance.a[j, k]
                    )

    for r in R:
        for j in N:
            if j != r:
                for k in K_e:
                    constraints.append(
                        instance.a[r, k] + instance.B_star / instance.e + instance.travel_time_matrix[r, j] * instance.x[r, j, k] <= instance.a[j, k]
                    )

    for k in K:
        constraints.append(instance.b[instance.O_prime, k] <= instance.time_window_end[0])

    for i in C:
        for k in K:
            constraints.append(instance.b[i, k] == instance.a[i, k] + instance.service_time[i])

    for r in R:
        for k in K_e:
            constraints.append(instance.b[r, k] == instance.a[r, k] + instance.B_star / instance.e)

    for i in N:
        for j in N:
            if j != i:
                for k in K_e:
                    constraints.append(
                        instance.B[i, k] - instance.L_ijk[i, j] * instance.x[i, j, k] + instance.B_star * (1 - instance.x[i, j, k]) >= instance.B[j, k]
                    )

    for k in K_e:
        constraints.append(instance.B[0, k] == instance.B_star)

    for r in R:
        for k in K_e:
            constraints.append(instance.b[r, k] >= instance.a[r, k] + instance.B_star / instance.e)

    for i in N:
        for k in K_e:
            constraints.append(instance.B[i, k] >= instance.B_star * instance.soc_min)

    for r in R:
        for k in K_e:
            constraints.append(instance.B[r, k] <= instance.B_star)

    return constraints

