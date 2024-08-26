import numpy as np

import copy


def read_solomon_instance(file_path):
    locations = []
    vehicles = {}

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('StringID'):
                    continue
                if line.startswith('Q Vehicle fuel tank capacity'):
                    vehicles['fuel_tank_capacity'] = float(line.split()[-1].replace('/', ''))
                elif line.startswith('C Vehicle load capacity'):
                    vehicles['load_capacity'] = float(line.split()[-1].replace('/', ''))
                elif line.startswith('r fuel consumption rate'):
                    vehicles['fuel_consumption_rate'] = float(line.split()[-1].replace('/', ''))
                elif line.startswith('g inverse refueling rate'):
                    vehicles['inverse_refueling_rate'] = float(line.split()[-1].replace('/', ''))
                elif line.startswith('v average Velocity'):
                    vehicles['average_velocity'] = float(line.split()[-1].replace('/', ''))
                else:
                    parts = line.split()
                    if len(parts) == 8:
                        try:
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
                        except ValueError as e:
                            print(f"Error parsing line: {line}. Error: {e}")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return locations, vehicles


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
        self.E_ijk = pi_e * self.F_ijk  # 确保这行定义了一个矩阵

        decay_transport = lambda t: 1 - np.exp(-self.theta_1 * t)
        decay_unload = lambda s: 1 - np.exp(-self.theta_2 * s)

        customer_demand_full = np.concatenate((self.customer_demand, np.zeros(self.N - self.n)))
        travel_time_matrix_customers = self.travel_time_matrix[:self.n, :self.n]

        time_window_start_full = np.concatenate((self.time_window_start, np.zeros(self.N - self.n)))
        time_window_end_full = np.concatenate((self.time_window_end, np.zeros(self.N - self.n)))

        C_31 = self.p_5 * np.sum(customer_demand_full[:self.n, None] * decay_transport(travel_time_matrix_customers))
        C_32 = self.p_5 * np.sum(
            customer_demand_full[:self.n, None] * decay_unload(np.zeros_like(travel_time_matrix_customers)))
        self.loss_cost = C_31 + C_32

        self.early_penalty = self.p_7 * np.sum(
            np.maximum(time_window_start_full[:self.n, None] - travel_time_matrix_customers, 0))
        self.late_penalty = self.p_8 * np.sum(
            np.maximum(travel_time_matrix_customers - time_window_end_full[:self.n, None], 0))
        self.time_window_penalty = self.early_penalty + self.late_penalty

        self.carbon_cost = self.c * np.sum(self.E_ijk[:self.m, :])  # 确保 E_ijk 是一个矩阵


    def calculate_fixed_cost(self):
        K_e = list(range(self.k_e))
        K_f = list(range(self.k_e, self.k_e + self.k_f))
        C1 = self.p_1 * sum(sum(self.x[0, j, k] for j in self.N) for k in K_e) + \
             self.p_2 * sum(sum(self.x[0, j, k] for j in self.N) for k in K_f)
        return C1

    def calculate_transport_cost(self):
        C2 = self.p_3 * sum(self.distance_matrix[i][j] * self.x[i, j, k]
                            for k in range(self.k_e)
                            for i in self.N
                            for j in self.N if i != j) + \
             self.p_4 * sum(self.distance_matrix[i][j] * self.x[i, j, k]
                            for k in range(self.k_e, self.k_e + self.k_f)
                            for i in self.N
                            for j in self.N if i != j)
        return C2

    def calculate_loss_cost(self):
        C3_1 = self.p_5 * sum(self.x[i, j, k] * self.customer_demand[j] *
                              (1 - np.exp(-self.theta_1 * (self.a[j, k] - self.b[0, k] + self.w[j, k])))
                              for i in self.N
                              for j in self.N if i != j
                              for k in range(self.k_e + self.k_f))

        C3_2 = self.p_5 * sum(self.x[i, j, k] * (self.u[i, j, k] - self.customer_demand[j]) *
                              (1 - np.exp(-self.theta_2 * self.service_time[j]))
                              for i in self.N
                              for j in self.N if i != j
                              for k in range(self.k_e + self.k_f))

        C3 = C3_1 + C3_2
        return C3

    def calculate_charging_cost(self):
        C4 = self.p_6 * sum(self.x[i, j, k] * self.f[i, j, k]
                            for i in self.N
                            for j in self.N if i != j
                            for k in range(self.k_e))
        return C4

    def calculate_time_window_penalty_cost(self):
        C5 = sum(self.p_7 * max(self.time_window_start[i] - self.a[i, k], 0) +
                 self.p_8 * max(self.a[i, k] - self.time_window_end[i], 0)
                 for i in self.C
                 for k in range(self.k_e + self.k_f))
        return C5

    def calculate_carbon_cost(self):
        C6 = self.c * sum(self.x[i, j, k] * self.E_ijk[i, j, k]
                          for i in self.N
                          for j in self.N if i != j
                          for k in range(self.k_e + self.k_f))
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
    def initialize_variables(self, k_e, k_f):
        self.k_e = k_e
        self.k_f = k_f
        self.x = np.zeros((self.N, self.N, self.k_e + self.k_f), dtype=int)
        self.u = np.zeros((self.N, self.N, self.k_e + self.k_f), dtype=float)
        self.a = np.zeros((self.N, self.k_e + self.k_f), dtype=float)
        self.b = np.zeros((self.N, self.k_e + self.k_f), dtype=float)
        self.B = np.zeros((self.N, self.k_e), dtype=float)
        self.f = np.zeros((self.N, self.N, self.k_e), dtype=float)

    def optimize_vehicle_numbers(self):
        k_e_bounds = (0, None)
        k_f_bounds = (0, None)

        def objective(vars):
            k_e, k_f = vars
            self.initialize_variables(int(k_e), int(k_f))
            fixed_cost = self.p_1 * k_e + self.p_2 * k_f
            return fixed_cost + self.transport_cost + self.loss_cost + self.charging_cost + self.time_window_penalty + self.carbon_cost

        initial_guess = [1, 1]

        result = minimize(
            objective,
            initial_guess,
            bounds=[k_e_bounds, k_f_bounds],
            method='SLSQP'
        )

        k_e_optimal, k_f_optimal = result.x
        self.initialize_variables(int(k_e_optimal), int(k_f_optimal))
        print(f"Optimal number of EVs: {int(k_e_optimal)}")
        print(f"Optimal number of conventional vehicles: {int(k_f_optimal)}")

    def print_objective(self):
        print(f"Objective Function Value: {self.objective}")

    def node_visit_constraints(self):
        constraints = []

        C = [i for i, loc in enumerate(self.locations) if loc['type'] == 'c']
        R = [i for i, loc in enumerate(self.locations) if loc['type'] == 'f']
        N = list(range(self.N))
        K = list(range(self.k_e + self.k_f))
        K_e = list(range(self.k_e))
        M = self.M

        constraints.append(
            sum(self.x[0, j, k] for j in C + R for k in K) == sum(self.x[j, self.O_prime, k] for j in C + R for k in K)
        )

        for i in C:
            constraints.append(
                sum(self.x[i, j, k] for j in N if j != i for k in K) == 1
            )

        for j in C:
            for k in K:
                constraints.append(
                    sum(self.x[i, j, k] for i in N if i != j) == sum(self.x[j, i, k] for i in N if i != j)
                )

        for r in R:
            for k in K_e:
                constraints.append(
                    sum(self.x[i, r, k] for i in N if i != r) <= 1
                )
                constraints.append(
                    sum(self.x[r, j, k] for j in N if j != r) <= 1
                )

        return constraints

    def load_balance_constraints(self):
        constraints = []

        C = [i for i, loc in enumerate(self.locations) if loc['type'] == 'c']
        R = [i for i, loc in enumerate(self.locations) if loc['type'] == 'f']
        N = list(range(self.N))
        E = [(i, j) for i in N for j in N if i != j]
        K = list(range(self.k_e + self.k_f))
        K_e = list(range(self.k_e))
        K_f = list(range(self.k_e, self.k_e + self.k_f))
        M = self.M
        Q_e = self.Q_e
        Q_f = self.Q_f
        q = self.customer_demand

        for (i, j) in E:
            for k in K_e:
                constraints.append(self.u[i, j, k] <= Q_e)
            for k in K_f:
                constraints.append(self.u[i, j, k] <= Q_f)

        for j in C:
            for k in K:
                constraints.append(
                    sum(self.u[i, j, k] for i in N if i != j) - sum(self.u[j, i, k] for i in N if i != j) + M * (
                                1 - sum(self.x[i, j, k] for i in N if i != j)) >= q[j]
                )

        for k in K:
            constraints.append(
                sum(self.u[i, self.O_prime, k] for i in C + R) == 0
            )

        return constraints

    def time_constraints(self):
        constraints = []

        C = [i for i, loc in enumerate(self.locations) if loc['type'] == 'c']
        R = [i for i, loc in enumerate(self.locations) if loc['type'] == 'f']
        N = list(range(self.N))
        K = list(range(self.k_e + self.k_f))
        M = self.M
        t = self.travel_time_matrix
        s = self.service_time
        w = np.zeros_like(s)
        E = [loc['ready_time'] for loc in self.locations]
        L = [loc['due_date'] for loc in self.locations]

        for k in K:
            constraints.append(self.b[0, k] >= E[0])

        for j in C:
            for k in K:
                constraints.append(
                    self.b[0, k] + t[0, j] * self.x[0, j, k] - M * (1 - self.x[0, j, k]) <= self.a[j, k]
                )

        for i in C:
            for j in N:
                if j != i:
                    for k in K:
                        constraints.append(
                            self.a[i, k] + t[i, j] * self.x[i, j, k] + s[i] + w[i, k] - M * (1 - self.x[i, j, k]) <=
                            self.a[j, k]
                        )

        for r in R:
            for k in K_e:
                constraints.append(
                    self.b[r, k] >= self.a[r, k] + self.B_star / self.e
                )

        for i in N:
            for k in K:
                constraints.append(
                    self.b[i, k] + t[i, self.O_prime] * self.x[i, self.O_prime, k] - M * (
                                1 - self.x[i, self.O_prime, k]) <= L[0]
                )

        for i in C:
            for k in K:
                constraints.append(
                    self.b[i, k] == self.a[i, k] + s[i] + w[i, k]
                )

        for i in R:
            for k in K_e:
                constraints.append(
                    self.b[i, k] == self.a[i, k] + M
                )

        for i in C:
            for k in K_e:
                constraints.append(
                    E[i] <= self.a[i, k] <= L[i]
                )

        return constraints

    def battery_constraints(self):
        constraints = []

        C = [i for i, loc in enumerate(self.locations) if loc['type'] == 'c']
        R = [i for i, loc in enumerate(self.locations) if loc['type'] == 'f']
        N = list(range(self.N))
        K_e = list(range(self.k_e))
        M = self.M
        L_ijk = self.L_ijk
        B_star = self.B_star
        soc_min = self.soc_min

        for i in N:
            for j in N:
                if j != i:
                    for k in K_e:
                        constraints.append(
                            self.B[i, k] - L_ijk[i, j] * self.x[i, j, k] + B_star * (1 - self.x[i, j, k]) >= self.B[
                                j, k]
                        )

        for k in K_e:
            constraints.append(
                self.B[0, k] == B_star
            )

        for i in N:
            for k in K_e:
                constraints.append(
                    self.B[i, k] >= B_star * soc_min
                )

        for r in R:
            for k in K_e:
                constraints.append(
                    self.B[r, k] == B_star
                )

        return constraints

    def variable_constraints(self):
        constraints = []

        N = list(range(self.N))
        K = list(range(self.k_e + self.k_f))

        for (i, j) in [(i, j) for i in N for j in N if i != j]:
            for k in K:
                constraints.append(self.u[i, j, k] >= 0)
                if k in range(self.k_e):
                    constraints.append(self.f[i, j, k] >= 0)
                constraints.append(self.x[i, j, k] == 0 or self.x[i, j, k] == 1)

        return constraints

    def additional_constraints(self):
        constraints = []

        C = [i for i, loc in enumerate(self.locations) if loc['type'] == 'c']
        R = [i for i, loc in enumerate(self.locations) if loc['type'] == 'f']
        N = list(range(self.N))
        K = list(range(self.k_e + self.k_f))
        K_e = list(range(self.k_e))
        M = self.M

        for k in K:
            constraints.append(self.b[0, k] >= self.time_window_start[0])

        for j in C:
            for k in K:
                constraints.append(
                    self.b[0, k] + self.travel_time_matrix[0, j] * self.x[0, j, k] <= self.a[j, k]
                )

        for i in C:
            for j in N:
                if j != i:
                    for k in K:
                        constraints.append(
                            self.a[i, k] + self.service_time[i] + self.travel_time_matrix[i, j] * self.x[i, j, k] <=
                            self.a[j, k]
                        )

        for r in R:
            for j in N:
                if j != r:
                    for k in K_e:
                        constraints.append(
                            self.a[r, k] + self.B_star / self.e + self.travel_time_matrix[r, j] * self.x[r, j, k] <=
                            self.a[j, k]
                        )

        for k in K:
            constraints.append(self.b[self.O_prime, k] <= self.time_window_end[0])

        for i in C:
            for k in K:
                constraints.append(self.b[i, k] == self.a[i, k] + self.service_time[i])

        for r in R:
            for k in K_e:
                constraints.append(self.b[r, k] == self.a[r, k] + self.B_star / self.e)

        for i in N:
            for j in N:
                if j != i:
                    for k in K_e:
                        constraints.append(
                            self.B[i, k] - self.L_ijk[i, j] * self.x[i, j, k] + self.B_star * (1 - self.x[i, j, k]) >=
                            self.B[j, k]
                        )

        for k in K_e:
            constraints.append(self.B[0, k] == self.B_star)

        for r in R:
            for k in K_e:
                constraints.append(self.b[r, k] >= self.a[r, k] + self.B_star / self.e)

        for i in N:
            for k in K_e:
                constraints.append(self.B[i, k] >= self.B_star * self.soc_min)

        for r in R:
            for k in K_e:
                constraints.append(self.B[r, k] <= self.B_star)

        return constraints

    def clustering(self):
        E, C = [], []
        barycentre_E = np.mean([loc for loc in self.locations if loc['type'] == 'c'], axis=0)
        barycentre_C = np.mean([loc for loc in self.locations if loc['type'] == 'c'], axis=0)
        d_E = [np.linalg.norm(np.array([loc['x'], loc['y']]) - barycentre_E) for loc in self.locations if
               loc['type'] == 'c']
        d_C = [np.linalg.norm(np.array([loc['x'], loc['y']]) - barycentre_C) for loc in self.locations if
               loc['type'] == 'c']
        d_E_min, d_E_max = min(d_E), max(d_E)
        d_C_min, d_C_max = min(d_C), max(d_C)
        q_min, q_max = min(self.customer_demand), max(self.customer_demand)

        unassigned_customers = list(range(self.n))

        while unassigned_customers:
            scores_E = [11 - 1 + (d - d_E_min) / (d_E_max - d_E_min) * 9 for d in d_E]
            scores_C = [11 - 1 + (d - d_C_min) / (d_C_max - d_C_min) * 9 * 0.5 + 11 - 1 + (q - q_min) / (
                        q_max - q_min) * 9 * 0.5 for d, q in zip(d_C, self.customer_demand)]

            max_score_E = max(scores_E)
            max_score_C = max(scores_C)
            i_star_E = scores_E.index(max_score_E)
            i_star_C = scores_C.index(max_score_C)

            if i_star_E == i_star_C:
                E.append(i_star_E)
                C.append(i_star_C)
            else:
                if max_score_E > max_score_C:
                    E.append(i_star_E)
                else:
                    C.append(i_star_C)

            barycentre_E = np.mean([self.locations[i] for i in E], axis=0)
            barycentre_C = np.mean([self.locations[i] for i in C], axis=0)
            d_E = [np.linalg.norm(np.array([loc['x'], loc['y']]) - barycentre_E) for loc in self.locations if
                   loc['type'] == 'c']
            d_C = [np.linalg.norm(np.array([loc['x'], loc['y']]) - barycentre_C) for loc in self.locations if
                   loc['type'] == 'c']

            unassigned_customers.remove(i_star_E if i_star_E in unassigned_customers else i_star_C)

        return E, C

    def calculate_best_position(self, route, customer):
        best_increase = float('inf')
        best_position = None
        for i in range(1, len(route)):
            increase = self.distance_matrix[route[i - 1], customer] + self.distance_matrix[customer, route[i]] - \
                       self.distance_matrix[route[i - 1], route[i]]
            if increase < best_increase:
                best_increase = increase
                best_position = i
        return best_position, best_increase

    def check_capacity_and_time(self, route, customer, is_electric):
        load = sum(self.customer_demand[loc - 1] for loc in route[1:-1])
        if load + self.customer_demand[customer - 1] > (self.Q_e if is_electric else self.Q_f):
            return False
        arrival_time = 0
        for i in range(1, len(route)):
            arrival_time += self.travel_time_matrix[route[i - 1], route[i]]
            if arrival_time > self.time_window_end[route[i] - 1]:
                return False
            arrival_time = max(arrival_time, self.time_window_start[route[i] - 1]) + self.service_time[route[i] - 1]
        return True

    def add_recharge_stations(self, route):
        new_route = [route[0]]
        battery = self.B_star
        for i in range(1, len(route)):
            distance = self.distance_matrix[route[i - 1], route[i]]
            if battery < distance:
                nearest_station = min(range(self.n, self.n + self.m),
                                      key=lambda s: self.distance_matrix[route[i - 1], s])
                new_route.append(nearest_station)
                battery = self.B_star
            new_route.append(route[i])
            battery -= distance
        return new_route

    def insertion_heuristic(self, cluster, is_electric):
        routes = []
        unserved = set(cluster)
        while unserved:
            route = [self.O]
            load = 0
            while unserved:
                best_increase = float('inf')
                best_position = None
                best_customer = None
                for customer in unserved:
                    position, increase = self.calculate_best_position(route, customer + 1)
                    if increase < best_increase:
                        best_increase = increase
                        best_position = position
                        best_customer = customer
                if load + self.customer_demand[best_customer] <= (self.Q_e if is_electric else self.Q_f):
                    route.insert(best_position, best_customer + 1)
                    load += self.customer_demand[best_customer]
                    unserved.remove(best_customer)
                else:
                    break
            route.append(self.O_prime)
            if is_electric:
                route = self.add_recharge_stations(route)
            routes.append(route)
        return routes

    def sequential_insertion_heuristic(self):
        E, C = self.clustering()

        eta_c = self.insertion_heuristic(C, False)
        eta_e = self.insertion_heuristic(E, True)

        eta = eta_c + eta_e
        return eta

    def print_solution(self, solution):
        for route in solution:
            print(" -> ".join(str(self.locations[i]['id']) for i in route))


class EVRPState:
    def __init__(self, routes, instance, unassigned=None):
        self.routes = routes
        self.instance = instance
        self.unassigned = unassigned if unassigned is not None else []
        self.load = [0] * len(routes)
        self.time = [0] * len(routes)
        self.battery = [instance.B_star] * len(routes)  # 假设所有路线开始时电池都是满的
        self.visited_customers = set()
        self.charging_station_visits = {i: 0 for i in range(len(instance.locations)) if
                                        instance.locations[i]['type'] == 'f'}

        for k, route in enumerate(routes):
            current_load = 0
            current_time = 0
            current_battery = self.battery[k] if k < instance.k_e else None
            for customer in route:
                if customer >= len(instance.locations):
                    raise IndexError(f"Customer index {customer} is out of bounds for locations array。")
                if instance.locations[customer]['type'] == 'c' and (customer - 1 >= len(instance.customer_demand)):
                    raise IndexError(f"Customer index {customer - 1} is out of bounds for customer demand array。")
                self.visited_customers.add(customer)
                if instance.locations[customer]['type'] == 'c':
                    demand = instance.customer_demand[customer]
                    current_load += demand
                if instance.locations[customer]['type'] == 'f':
                    if self.charging_station_visits[customer] < 2:
                        self.charging_station_visits[customer] += 1
                        current_battery = instance.B_star  # 重置电池电量
                    else:
                        raise ValueError(f"Charging station {customer} has been visited more than twice。")
                self.load[k] = current_load
                if k < instance.k_e:
                    current_battery -= self.calculate_battery_usage(k, customer)
                    self.battery[k] = current_battery
                current_time += self.calculate_travel_time(k, customer)
                self.time[k] = current_time

    def copy(self):
        return EVRPState(copy.deepcopy(self.routes), self.instance, self.unassigned.copy())

    def objective(self):
        return sum(self.route_cost(route) for route in self.routes)

    @property
    def cost(self):
        return self.objective()

    def find_route(self, customer):
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Solution does not contain customer {customer}。")

    def route_cost(self, route):
        distances = self.instance.distance_matrix
        tour = [self.instance.O] + route + [self.instance.O_prime]
        return sum(distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))

    def update_state(self, route_index, customer, load, time, battery=None):
        self.visited_customers.add(customer)
        if self.instance.locations[customer]['type'] == 'f':
            if self.charging_station_visits[customer] < 2:
                self.charging_station_visits[customer] += 1
                battery = self.instance.B_star  # 重置电池电量
            else:
                raise ValueError(f"Charging station {customer} has been visited more than twice。")
        self.load[route_index] = load
        self.time[route_index] = time
        if battery is not None:
            self.battery[route_index] = battery

    def get_load(self, route_index):
        return self.load[route_index]

    def get_time(self, route_index):
        return self.time[route_index]

    def get_battery(self, route_index):
        return self.battery[route_index] if route_index < self.instance.k_e else None

    def add_unassigned(self, customer):
        self.unassigned.append(customer)

    def remove_unassigned(self, customer):
        self.unassigned.remove(customer)

    def is_unassigned(self, customer):
        return customer in self.unassigned

    def calculate_travel_time(self, vehicle, customer):
        if len(self.routes[vehicle]) == 0:
            prev_location = self.instance.O
        else:
            prev_location = self.routes[vehicle][-1]
        return self.instance.travel_time_matrix[prev_location][customer]

    def calculate_battery_usage(self, vehicle, customer):
        if len(self.routes[vehicle]) == 0:
            prev_location = self.instance.O
        else:
            prev_location = self.routes[vehicle][-1]
        return self.instance.L_ijk[prev_location][customer]

    def is_customer_visited(self, customer):
        return customer in self.visited_customers

    def validate_route(self, route_index):
        load = 0
        time = 0
        battery = self.battery[route_index] if route_index < self.instance.k_e else None

        for customer in self.routes[route_index]:
            if customer >= len(self.instance.locations):
                raise IndexError(f"Customer index {customer} is out of bounds for locations array。")
            if self.instance.locations[customer]['type'] == 'c' and (
                    customer - 1 >= len(self.instance.customer_demand)):
                raise IndexError(f"Customer index {customer - 1} is out of bounds for customer demand array。")
            if self.instance.locations[customer]['type'] == 'c':
                demand = self.instance.customer_demand[customer]
                load += demand
            if load > self.instance.Q_e and route_index < self.instance.k_e:
                raise ValueError(f"Route {route_index} exceeds electric vehicle load capacity。")
            if load > self.instance.Q_f and route_index >= self.instance.k_e:
                raise ValueError(f"Route {route_index} exceeds fuel vehicle load capacity。")

            if route_index < self.instance.k_e:
                battery -= self.calculate_battery_usage(route_index, customer)
                if battery < self.instance.B_star * self.instance.soc_min:
                    raise ValueError(f"Route {route_index} violates battery constraints。")
                self.battery[route_index] = battery

            if self.instance.locations[customer]['type'] == 'f':
                if self.charging_station_visits[customer] >= 2:
                    raise ValueError(f"Charging station {customer} has been visited more than twice。")
                self.charging_station_visits[customer] += 1
                battery = self.instance.B_star  # 重置电池电量

            time += self.calculate_travel_time(route_index, customer)
            if time > self.instance.time_window_end[customer]:
                raise ValueError(f"Route {route_index} violates time window constraints。")
            self.time[route_index] = time
            self.load[route_index] = load

        return True


if __name__ == "__main__":
    file_path = "D:\\2024\\ZSY-ALNS\\pythonProject1\\evrptw_instances\\c101_21.txt"
    locations, vehicles = read_solomon_instance(file_path)
    instance = CustomEVRPInstance(locations, vehicles)

    instance.optimize_vehicle_numbers()
    solution = instance.sequential_insertion_heuristic()
    instance.print_solution(solution)
