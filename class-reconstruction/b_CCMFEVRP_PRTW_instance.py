from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from a_read_instance import read_solomon_instance

@dataclass
class Location:
    id: str
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float
    charging_rate: float = 3.39  # 默认充电速率为3.39，只有充电桩才有意义


@dataclass
class CustomerLocation(Location):
    type: str = 'c'

@dataclass
class ChargingStationLocation(Location):
    charging_rate: float = 3.39
    type: str = 'f'

@dataclass
class DepotLocation(Location):
    def __init__(self, id: str, x: float, y: float):
        super().__init__(id, x, y, demand=0, ready_time=0, due_date=float('inf'), service_time=0)

@dataclass
class CustomEVRPInstance:
    locations: List[Location]
    vehicles: Dict[str, float]
    location_id_to_index: Dict[str, int] = field(init=False)
    customer_indices: List[int] = field(init=False)
    charging_station_indices: List[int] = field(init=False)
    depot_index: int = field(init=False)
    n: int = field(init=False)
    m: int = field(init=False)
    O: Location = field(init=False)
    O_prime: int = field(init=False)
    N: int = field(init=False)
    Q_e: float = field(init=False)
    Q_f: float = field(init=False)
    B_star: float = 40
    v_e: float = field(init=False)
    v_f: float = field(init=False)
    m_v_e: float = 2000
    m_v_f: float = 2500
    e: float = field(init=False)
    p_1: float = 500
    p_2: float = 400
    p_3: float = 2
    p_4: float = 5
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
        self.customer_indices = [idx for idx, loc in enumerate(self.locations) if isinstance(loc, CustomerLocation)]
        self.charging_station_indices = [idx for idx, loc in enumerate(self.locations) if isinstance(loc, ChargingStationLocation)]
        self.depot_index = next(idx for idx, loc in enumerate(self.locations) if isinstance(loc, DepotLocation))
        self.type_to_indices = {
            'c': self.customer_indices,
            'f': self.charging_station_indices,
            'd': [self.depot_index]
        }

        self.n = len(self.customer_indices)
        self.m = len(self.charging_station_indices)
        self.O = self.locations[self.depot_index]
        self.O_prime = self.n + self.m + 1
        self.N = self.n + self.m + 2

        self.Q_e = self.vehicles.get('load_capacity', 2000)
        self.Q_f = self.vehicles.get('load_capacity', 2500)
        self.v_e = self.vehicles.get('average_velocity', 60)
        self.v_f = self.vehicles.get('average_velocity', 50)
        self.e = self.vehicles.get('inverse_refueling_rate', 0.1)
        # self.B_star = self.vehicles.get('fuel_tank_capacity',20 )

        self.energy_consumption_rate_electric = self.vehicles.get('electric_consumption_rate', 0.2)
        self.energy_consumption_rate_fuel = self.vehicles.get('fuel_consumption_rate', 0.3)

        self.q_i = np.array([loc.demand for loc in self.locations])
        self.E_i = np.array([loc.ready_time for loc in self.locations])
        self.L_i = self.E_i + np.array([loc.due_date for loc in self.locations])

        coords = np.array([(loc.x, loc.y) for loc in self.locations])
        self.d_ij = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        self.t_ijk_e = self.d_ij / self.v_e
        self.t_ijk_f = self.d_ij / self.v_f

        self.initialize_decision_variables()
        self.precomputed_f_ijk = np.maximum(0, self.B_star - self.B_ik1)

        self.customer_positions = np.array([(loc.x, loc.y) for loc in self.locations if isinstance(loc, CustomerLocation)])
        self.depot_position = np.array([self.O.x, self.O.y])
        for loc in self.locations:
            if isinstance(loc, ChargingStationLocation):
                loc.charging_rate = self.vehicles.get('charging_rate', 3.39)
                print(f"Charging Station ID: {loc.id}, Charging Rate: {loc.charging_rate}")

        self.calculate_energy_consumption_factors()

    def initialize_decision_variables(self):
        self.x_ijk = np.zeros((self.N, self.N, self.K_e + self.K_f), dtype=int)
        self.u_ijk = np.zeros((self.N, self.N, self.K_e + self.K_f))
        self.f_ijk = np.zeros((self.N, self.N, self.K_e))
        self.B_ik1 = np.full((self.N, self.K_e), self.B_star)
        self.B_ik2 = np.full((self.N, self.K_e), self.B_star)
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

        v_e_m_per_s = self.v_e / 3.6
        K_ijk_e = 0.5 * c_d * rho * A * v_e_m_per_s ** 3 + (self.m_v_e + self.q_i.mean()) * g * c_d * v_e_m_per_s
        L_ijk_e_joules = phi_d * varphi_d * K_ijk_e * (self.d_ij * 1000)

        self.L_ijk_e = L_ijk_e_joules * 2.77778e-7

        xi = 14.7
        kappa = 44.4
        psi = 0.8
        sigma = 0.3
        theta = 1000
        omega = 2.0
        eta = 0.3
        tau = 0.85

        K_ijk_f = K_ijk_e
        F_ijk_f_joules = (xi / (kappa * psi)) * (sigma * theta * omega + K_ijk_f / (eta * tau)) * (self.d_ij * 1000)

        self.F_ijk_f = F_ijk_f_joules * 1e-6

    def calculate_charging_time(self, charging_station_idx, current_battery, target_battery):
        charging_station = self.locations[charging_station_idx]
        charging_rate = charging_station.charging_rate
        charging_time = (target_battery - current_battery) / charging_rate
        return charging_time

    def update_vehicle_numbers(self, new_K_e: int, new_K_f: int):
        self.K_e = new_K_e
        self.K_f = new_K_f

        self.initialize_decision_variables()
        self.calculate_energy_consumption_factors()

    def calculate_energy_consumption(self, current_location, next_location, vehicle_type):
        if current_location >= len(self.d_ij) or next_location >= len(self.d_ij):
            raise IndexError("Location index out of bounds")

        distance = self.d_ij[current_location, next_location]
        if vehicle_type == 'electric':
            return distance * self.energy_consumption_rate_electric
        else:
            return distance * self.energy_consumption_rate_fuel

    def calculate_battery_usage(self, route_in):
        battery_usage = 0.0
        for i in range(len(route_in) - 1):
            start = route_in[i]
            end = route_in[i + 1]
            battery_usage += self.L_ijk_e[start, end]
        return np.clip(battery_usage, 0, self.B_star)

    def calculate_fuel_consumption(self, route_in):
        total_fuel = 0
        for i in range(len(route_in) - 1):
            total_fuel += self.F_ijk_f[route_in[i], route_in[i + 1]]
        return np.clip(total_fuel, 0, self.vehicles['fuel_tank_capacity'])

    def calculate_total_cost(self, route_in, vehicle_type: str) -> float:
        fixed_cost = self.calculate_fixed_cost(vehicle_type)
        transport_cost = self.calculate_transport_cost(route_in, vehicle_type)
        loss_cost = self.calculate_loss_cost(route_in, vehicle_type)
        charging_cost = 0.0
        if vehicle_type == 'electric':
            charging_cost = self.calculate_charging_cost(route_in, vehicle_type)
        time_window_penalty = self.calculate_time_window_penalty(route_in, vehicle_type)
        carbon_cost = self.calculate_carbon_cost(route_in, vehicle_type)

        return fixed_cost + transport_cost + loss_cost + charging_cost + time_window_penalty + carbon_cost

    def calculate_fixed_cost(self, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return self.p_1 * self.K_e
        elif vehicle_type == 'fuel':
            return self.p_2 * self.K_f
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_transport_cost(self, route_in, vehicle_type: str) -> float:
        if len(route_in) < 2:
            return 0.0

        route_array = np.array(route_in)

        if vehicle_type == 'electric':
            return self.p_3 * np.sum(self.d_ij[route_array[:-1], route_array[1:]])
        elif vehicle_type == 'fuel':
            return self.p_4 * np.sum(self.d_ij[route_array[:-1], route_array[1:]])
        else:
            raise ValueError("Unknown vehicle type.")

    def calculate_time_satisfaction(self, customer):
        adjusted_customer = customer
        if adjusted_customer < 0 or adjusted_customer >= len(self.E_i):
            raise IndexError(f"Customer index {customer} is out of bounds for E_i with size {len(self.E_i)}.")

        ready_time = self.E_i[adjusted_customer]
        due_date = self.L_i[adjusted_customer]

        satisfaction = due_date - ready_time

        return satisfaction

    def calculate_loss_cost(self, route_in, vehicle_type: str) -> float:
        total_loss_cost = 0.0

        if vehicle_type == 'electric' or vehicle_type == 'fuel':
            for i in range(len(route_in) - 1):
                j = route_in[i + 1]
                if j < 0 or j >= len(self.q_i):
                    continue

                for k in range(self.K_e if vehicle_type == 'electric' else self.K_f):
                    x_ijk = self.x_ijk[route_in[i], route_in[i + 1], k]
                    q_j = self.q_i[j]
                    a_jk = self.a_ik[j, k]
                    b_O = self.b_ik[self.depot_index, k]
                    w_jk = self.w_ik[j, k]
                    u_ijk = self.u_ijk[route_in[i], route_in[i + 1], k]

                    if vehicle_type == 'electric':
                        s_j = self.L_ijk_e[route_in[i], route_in[i + 1]]
                    else:
                        s_j = self.F_ijk_f[route_in[i], route_in[i + 1]]

                    C_31 = self.p_5 * x_ijk * q_j * (1 - np.exp(-self.theta_1 * (a_jk - b_O + w_jk)))
                    C_32 = self.p_5 * x_ijk * (u_ijk - q_j) * (1 - np.exp(-self.theta_2 * s_j))

                    total_loss_cost += C_31 + C_32

        return total_loss_cost

    def calculate_charging_cost(self, route_in, vehicle_type: str) -> float:
        if vehicle_type == 'fuel':
            return 0.0
        total_cost = 0.0
        chunk_size = 5

        for i in range(len(route_in) - 1):
            current_node = route_in[i]
            next_node = route_in[i + 1]
            for j in range(0, self.B_ik1.shape[1], chunk_size):
                f_ijk_chunk = np.maximum(0, self.B_star - self.B_ik1[current_node, j:j + chunk_size])
                total_cost += np.sum(f_ijk_chunk)

        return self.p_6 * total_cost

    def calculate_time_window_penalty(self, route_in, vehicle_type: str) -> float:
        if len(route_in) < 2:
            return 0.0

        current_time = 0
        total_penalty = 0.0

        for i in range(1, len(route_in)):
            start = route_in[i - 1]
            end = route_in[i]

            travel_time = self.t_ijk_e[start, end] if vehicle_type == 'electric' else self.t_ijk_f[start, end]
            current_time += travel_time

            ready_time = self.E_i[end]
            due_date = self.L_i[end]

            if current_time < ready_time:
                current_time = ready_time

            if current_time > due_date:
                total_penalty += (current_time - due_date) * self.p_7

            service_time = self.locations[end].service_time
            current_time += service_time

        return total_penalty

    def calculate_carbon_cost(self, route_in, vehicle_type: str) -> float:
        if vehicle_type == 'electric':
            return 0.0
        elif vehicle_type == 'fuel':
            pi_e = 2.31

            driving_segments = [(route_in[i], route_in[i + 1]) for i in range(len(route_in) - 1)]
            carbon_emissions = 0
            for start, end in driving_segments:
                carbon_emissions += pi_e * self.F_ijk_f[start, end]

            total_cost = self.c * carbon_emissions
            return total_cost
        else:
            raise ValueError("Unknown vehicle type.")

    def print_customer_info(self):
        print("Customer Information:")
        for idx in self.customer_indices:
            loc = self.locations[idx]
            print(f"Customer Index: {idx}, ID: {loc.id}, Coordinates: ({loc.x}, {loc.y}), Demand: {loc.demand}, Ready Time: {loc.ready_time}, Due Date: {loc.due_date}")

    def print_charging_station_info(self):
        print("\nCharging Station Information:")
        for idx in self.charging_station_indices:
            loc = self.locations[idx]
            print(f"Charging Station Index: {idx}, ID: {loc.id}, Coordinates: ({loc.x}, {loc.y}), Charging Rate: {loc.charging_rate}")

if __name__ == "__main__":
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = []

    for loc in locations_data:
        if loc['type'] == 'c':
            locations.append(CustomerLocation(**loc))
        elif loc['type'] == 'f':
            locations.append(ChargingStationLocation(**loc))
        elif loc['type'] == 'd':
            locations.append(DepotLocation(id=loc['id'], x=loc['x'], y=loc['y']))

    instance = CustomEVRPInstance(locations, vehicles_data)

    # 打印客户和充电站信息
    instance.print_customer_info()
    instance.print_charging_station_info()

    # 使用提供的电动车和燃油车路径进行成本计算
    electric_route = [0, 54, 59, 116, 118, 114, 113, 5, 0]
    fuel_route = [0, 88, 86, 87, 90, 83, 82, 85, 89, 93, 61, 62, 63, 65, 66, 67, 69, 0]

    # 调试电动车成本计算
    print("\n--- Electric Vehicle Cost Calculation ---")
    fixed_cost_electric = instance.calculate_fixed_cost('electric')
    print(f"Fixed Cost (Electric): {fixed_cost_electric}")

    transport_cost_electric = instance.calculate_transport_cost(electric_route, 'electric')
    print(f"Transport Cost (Electric): {transport_cost_electric}")

    loss_cost_electric = instance.calculate_loss_cost(electric_route, 'electric')
    print(f"Loss Cost (Electric): {loss_cost_electric}")

    charging_cost_electric = instance.calculate_charging_cost(electric_route, 'electric')
    print(f"Charging Cost (Electric): {charging_cost_electric}")

    time_window_penalty_electric = instance.calculate_time_window_penalty(electric_route, 'electric')
    print(f"Time Window Penalty (Electric): {time_window_penalty_electric}")

    carbon_cost_electric = instance.calculate_carbon_cost(electric_route, 'electric')
    print(f"Carbon Cost (Electric): {carbon_cost_electric}")

    total_cost_electric = instance.calculate_total_cost(electric_route, 'electric')
    print(f"Total Cost (Electric): {total_cost_electric}")

    # 调试燃油车成本计算
    print("\n--- Fuel Vehicle Cost Calculation ---")
    fixed_cost_fuel = instance.calculate_fixed_cost('fuel')
    print(f"Fixed Cost (Fuel): {fixed_cost_fuel}")

    transport_cost_fuel = instance.calculate_transport_cost(fuel_route, 'fuel')
    print(f"Transport Cost (Fuel): {transport_cost_fuel}")

    loss_cost_fuel = instance.calculate_loss_cost(fuel_route, 'fuel')
    print(f"Loss Cost (Fuel): {loss_cost_fuel}")

    charging_cost_fuel = instance.calculate_charging_cost(fuel_route, 'fuel')
    print(f"Charging Cost (Fuel): {charging_cost_fuel}")  # 燃油车的充电成本应为0

    time_window_penalty_fuel = instance.calculate_time_window_penalty(fuel_route, 'fuel')
    print(f"Time Window Penalty (Fuel): {time_window_penalty_fuel}")

    carbon_cost_fuel = instance.calculate_carbon_cost(fuel_route, 'fuel')
    print(f"Carbon Cost (Fuel): {carbon_cost_fuel}")

    total_cost_fuel = instance.calculate_total_cost(fuel_route, 'fuel')
    print(f"Total Cost (Fuel): {total_cost_fuel}")
