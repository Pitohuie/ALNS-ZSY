from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance
import copy  # Import the copy module


@dataclass
class State:
    current_route: List[int]
    total_time: float
    total_cost: float
    total_energy_consumption: float
    total_fuel_consumption: float
    vehicle_type: str
    satisfied_constraints: bool

    def copy(self):
        return copy.deepcopy(self)  # Use deepcopy to ensure a full copy of the state

    def objective(self) -> float:
        """
        This function should return the total cost as the objective function value.
        """
        return self.total_cost


class ALNSStateManager:
    def __init__(self, instance: CustomEVRPInstance, constraints):
        self.instance = instance
        self.constraints = constraints

    def initialize_state(self, route: List[int], vehicle_type: str) -> State:
        total_time = self.calculate_total_time(route, vehicle_type)
        total_cost = self.instance.calculate_total_cost(route, vehicle_type)
        total_energy_consumption = self.instance.calculate_energy_consumption(route) \
            if vehicle_type == 'electric' else 0.0
        total_fuel_consumption = self.instance.calculate_fuel_consumption(route) if vehicle_type == 'fuel' else 0.0

        satisfied_constraints = self.constraints.check_node_visit(route) and \
                                self.constraints.check_load_balance(route, vehicle_type) and \
                                self.constraints.check_time_window(route)

        return State(
            current_route=route,
            total_time=total_time,
            total_cost=total_cost,
            total_energy_consumption=total_energy_consumption,
            total_fuel_consumption=total_fuel_consumption,
            vehicle_type=vehicle_type,
            satisfied_constraints=satisfied_constraints
        )

    def calculate_total_time(self, route: List[int], vehicle_type: str) -> float:
        current_time = 0
        for i in range(1, len(route)):
            start = route[i - 1]
            end = route[i]
            travel_time = self.instance.t_ijk_e[start, end] \
                if vehicle_type == 'electric' else self.instance.t_ijk_f[start, end]
            current_time += travel_time
            service_time = self.instance.locations[end].service_time
            current_time += service_time
        return current_time

    def update_state(self, state: State, route: List[int]) -> State:
        state.current_route = route
        state.total_time = self.calculate_total_time(route, state.vehicle_type)
        state.total_cost = self.instance.calculate_total_cost(route, state.vehicle_type)
        state.total_energy_consumption = self.instance.calculate_energy_consumption(route) \
            if state.vehicle_type == 'electric' else 0.0
        state.total_fuel_consumption = self.instance.calculate_fuel_consumption(route) \
            if state.vehicle_type == 'fuel' else 0.0
        state.satisfied_constraints = self.constraints.check_node_visit(route) and \
                                      self.constraints.check_load_balance(route, state.vehicle_type) and \
                                      self.constraints.check_time_window(route)
        return state
