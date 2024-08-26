# Starting with step 1: Modifying the `b_CCMFEVRP_PRTW_instance.py` to enhance vehicle number update and energy management

# Since there are multiple changes to make, I'll approach each section step by step, beginning with dynamic vehicle count updates.
# This will be followed by more robust energy management, battery constraints, and associated parameter adjustments.

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

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
    K_e: int = 1  # Updated to initialize with 1 as a dynamic variable
    K_f: int = 1  # Updated to initialize with 1 as a dynamic variable
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
    minimum_battery_threshold: float = 20  # Assumed value, can be adjusted based on reality

    def __post_init__(self):
        # Initialize location and index mappings
        self.location_id_to_index = {loc.id: idx for idx, loc in enumerate(self.locations)}
        self.n = sum(1 for loc in self.locations if loc.type == 'c')
        self.m = sum(1 for loc in self.locations if loc.type == 'f')
        self.O = next(loc for loc in self.locations if loc.type == 'd')
        self.O_prime = self.n + self.m + 1
        self.N = self.n + self.m + 2

        # Initialize vehicle capacities and parameters
        self.Q_e = self.vehicles['load_capacity']
        self.Q_f = self.vehicles['load_capacity']
        self.v_e = self.vehicles['average_velocity']
        self.v_f = self.vehicles['average_velocity']
        self.e = self.vehicles['inverse_refueling_rate']

        # Initialize demands, time windows
        self.q_i = np.array([loc.demand for loc in self.locations if loc.type == 'c'])
        self.E_i = np.array([loc.ready_time for loc in self.locations if loc.type == 'c'])
        self.L_i = self.E_i + np.array([loc.due_date for loc in self.locations if loc.type == 'c'])

        # Initialize distance and time matrices
        coords = np.array([(loc.x, loc.y) for loc in self.locations])
        self.d_ij = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        self.t_ijk_e = self.d_ij / self.v_e
        self.t_ijk_f = self.d_ij / self.v_f
        self.initialize_decision_variables()

        # Precompute fuel cost
        self.precomputed_f_ijk = np.maximum(0, self.B_star - self.B_ik1)

        # Initialize positions
        self.customer_positions = np.array([(loc.x, loc.y) for loc in self.locations if loc.type == 'c'])
        depot = next(loc for loc in self.locations if loc.type == 'd')
        self.depot_position = np.array([depot.x, depot.y])

        # Energy consumption calculations for electric and fuel vehicles
        self.calculate_energy_consumption_factors()

    def initialize_decision_variables(self):
        # Initialize decision variables based on vehicle counts
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
        # Common parameters
        c_d = 0.3  # Drag coefficient
        rho = 1.225  # Air density (kg/m^3)
        A = 2.5  # Vehicle frontal area (m^2)
        g = 9.81  # Gravity acceleration (m/s^2)
        phi_d = 0.9  # Aerodynamic efficiency coefficient
        varphi_d = 0.85  # Vehicle powertrain efficiency coefficient

        ### Energy consumption for electric vehicles ###
        K_ijk_e = 0.5 * c_d * rho * A * self.v_e ** 3 + (self.m_v_e + self.q_i.mean()) * g * c_d * self.v_e
        self.L_ijk_e = phi_d * varphi_d * K_ijk_e * self.t_ijk_e

        ### Energy consumption for fuel vehicles ###
        xi = 14.7  # Fuel calorific value coefficient
        kappa = 44.8  # Fuel efficiency coefficient
        psi = 0.85  # Engine efficiency coefficient
        sigma = 0.5  # Fuel efficiency coefficient
        vartheta = 30  # Engine power coefficient
        omega = 2.0  # Engine speed coefficient
        eta = 0.3  # Transmission efficiency
        tau = 0.9  # Overall efficiency coefficient

        K_ijk_f = 0.5 * c_d * rho * A * self.v_f ** 3 + (self.m_v_f + self.q_i.mean()) * g * c_d * self.v_f
        self.F_ijk_f = (xi / (kappa * psi)) * (sigma * vartheta * omega + K_ijk_f / (eta * tau))
        self.F_ijk_f = np.full_like(self.d_ij, self.F_ijk_f)

        # Fuel vehicles have carbon emissions based on F_ijk_f
        pi_e = 2.31  # Unit energy price for electric vehicles
        self.E_ijk_f = pi_e * self.F_ijk_f

    def update_vehicle_numbers(self, new_K_e: int, new_K_f: int):
        """
        Dynamically update the number of electric and fuel vehicles,
        and update all related decision variables and parameters.
        """
        self.K_e = new_K_e
        self.K_f = new_K_f

        # Re-initialize decision variables with new vehicle numbers
        self.initialize_decision_variables()

        # Update any other related parameters if necessary
        self.calculate_energy_consumption_factors()

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
            total_energy += self.L_ijk_e[start, end]  # 这里需要使用事先计算好的能量消耗矩阵
        return total_energy

