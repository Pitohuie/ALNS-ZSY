from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from numpy.random import RandomState
from typing import List, Tuple
from destroy_operators import DestructionOperators
from repair_operators import InsertionOperators
from a_read_instance import read_solomon_instance
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from c_constraints import Constraints
from e_nnwithchargingstation import construct_initial_solution
from alns.stop import MaxIterations
import matplotlib.pyplot as plt
import numpy as np
import itertools

# 停止条件：在100次迭代后停止
stop_criterion = MaxIterations(100)

class Solution(State):
    def __init__(self, instance: CustomEVRPInstance, electric_routes: List[Tuple[List[int], float]],
                 fuel_routes: List[Tuple[List[int], float]]):
        self.instance = instance
        self.electric_routes = electric_routes
        self.fuel_routes = fuel_routes
        self.objective_value = self.calculate_total_cost()
        self.battery_levels = [self.instance.B_star] * len(electric_routes)  # 初始化每条电动车路线的电量
        self.constraints = Constraints(instance)
        self.visited_charging_stations = [False] * len(electric_routes)  # 标记每条电动车路线是否已经访问过充电桩

    def sync_battery_levels(self):
        """确保 battery_levels 和 electric_routes 长度保持一致"""
        if len(self.battery_levels) > len(self.electric_routes):
            self.battery_levels = self.battery_levels[:len(self.electric_routes)]
        elif len(self.battery_levels) < len(self.electric_routes):
            self.battery_levels.extend([self.instance.B_star] * (len(self.electric_routes) - len(self.battery_levels)))

    def sync_visited_stations(self):
        """确保 visited_charging_stations 和 electric_routes 长度保持一致"""
        if len(self.visited_charging_stations) > len(self.electric_routes):
            self.visited_charging_stations = self.visited_charging_stations[:len(self.electric_routes)]
        elif len(self.visited_charging_stations) < len(self.electric_routes):
            self.visited_charging_stations.extend([False] * (len(self.electric_routes) - len(self.visited_charging_stations)))

    def add_customer(self, route_index: int, position: int, customer: int) -> None:
        self.sync_battery_levels()  # 每次操作前同步 battery_levels
        self.sync_visited_stations()  # 每次操作前同步 visited_charging_stations

        if route_index >= len(self.electric_routes + self.fuel_routes):
            raise IndexError(f"Route index {route_index} is out of range.")

        if route_index < len(self.electric_routes):
            self._add_customer_to_electric_route(route_index, position, customer)
        else:
            self._add_customer_to_fuel_route(route_index - len(self.electric_routes), position, customer)

    def _add_customer_to_electric_route(self, route_index: int, position: int, customer: int):
        if route_index >= len(self.electric_routes):
            raise IndexError(f"Electric route index {route_index} is out of range.")

        route = self.electric_routes[route_index][0]
        if position >= len(route):
            raise IndexError(f"Position {position} is out of range for route with length {len(route)}.")

        route = self.electric_routes[route_index][0]
        if self.is_charging_station(customer):
            if self.visited_charging_stations[route_index]:
                self.return_to_depot(route_index)
                self.electric_routes[route_index] = ([self.instance.depot_index, customer], 0)
            else:
                self.visited_charging_stations[route_index] = True
                route.insert(position, customer)
        else:
            previous_customer = route[-1]
            energy_consumption = self.instance.calculate_energy_consumption(previous_customer, customer, 'electric')
            if self.battery_levels[route_index] < energy_consumption:
                self.return_to_depot(route_index)
                self.electric_routes[route_index][0].append(customer)
                self.update_battery_level(route_index, energy_consumption)
            else:
                self.update_battery_level(route_index, energy_consumption)
                route.insert(position, customer)


    def remaining_battery(self, route_index: int):
        """返回指定路径的剩余电量"""
        if 0 <= route_index < len(self.battery_levels):
            return self.battery_levels[route_index]
        raise IndexError(f"Invalid route_index {route_index}. Must be within range.")

    def update_battery_level(self, route_index, energy_consumption):
        """更新指定路径的剩余电量"""
        if 0 <= route_index < len(self.battery_levels):
            self.battery_levels[route_index] -= energy_consumption
        else:
            raise IndexError(f"Attempted to update battery level for a route with invalid index {route_index}.")



    def _add_customer_to_fuel_route(self, route_index: int, position: int, customer: int):
        if route_index >= len(self.fuel_routes):
            raise IndexError(f"Fuel route index {route_index} is out of range.")
        self.fuel_routes[route_index][0].insert(position, customer)

    def return_to_depot(self, route_index: int) -> None:
        """将车辆路径返回到配送中心，并重新开始新路线"""
        if route_index < len(self.electric_routes):
            route = self.electric_routes[route_index][0]
            depot_index = self.instance.depot_index
            route.append(depot_index)
            # 重置充电桩访问标记和电量
            self.visited_charging_stations[route_index] = False
            self.battery_levels[route_index] = self.instance.B_star
        else:
            raise IndexError(f"Attempted to return a non-electric vehicle route to the depot with index {route_index}.")

    def is_charging_station(self, customer: int) -> bool:
        """检查是否为充电桩"""
        return customer in self.instance.charging_station_indices

    def is_feasible(self):
        """检查当前解是否在所有约束下可行"""
        for route, _ in self.electric_routes:
            if not self.constraints.check_node_visit(route) or \
               not self.constraints.check_load_balance(route, 'electric') or \
               not self.constraints.check_time_window(route) or \
               not self.constraints.check_battery(route):
                return False

        for route, _ in self.fuel_routes:
            if not self.constraints.check_node_visit(route) or \
               not self.constraints.check_load_balance(route, 'fuel') or \
               not self.constraints.check_time_window(route):
                return False

        return True

    def copy(self) -> 'Solution':
        copied_solution = super().copy()
        copied_solution.battery_levels = self.battery_levels.copy()
        copied_solution.visited_charging_stations = self.visited_charging_stations.copy()  # 确保复制时保留充电桩访问信息
        if not copied_solution.is_feasible():
            raise ValueError("Copied solution is not feasible.")
        return copied_solution

    def calculate_total_cost(self) -> float:
        total_cost = 0.0
        for route_tuple in self.electric_routes:
            route = route_tuple[0]
            if isinstance(route, list) and len(route) > 1:
                route_cost = self.instance.calculate_total_cost(route, 'electric')
                total_cost += route_cost

        for route_tuple in self.fuel_routes:
            route = route_tuple[0]
            if isinstance(route, list) and len(route) > 1:
                route_cost = self.instance.calculate_total_cost(route, 'fuel')
                total_cost += route_cost

        return total_cost

    def remove_customer(self, customer: int) -> None:
        for route in self.electric_routes + self.fuel_routes:
            if customer in route[0]:
                route[0].remove(customer)
                break

    def objective(self) -> float:
        return self.calculate_total_cost()

    def get_route_index(self, customer: int) -> int:
        for idx, route in enumerate(self.electric_routes + self.fuel_routes):
            if customer in route[0]:
                return idx
        return -1  # 如果客户未找到，返回 -1

    def get_position(self, customer: int) -> int:
        for route in self.electric_routes + self.fuel_routes:
            if customer in route[0]:
                return route[0].index(customer)
        return -1  # 如果客户未找到，返回 -1

    def __len__(self):
        return len(self.electric_routes) + len(self.fuel_routes)

    def plot_routes_with_stations(self, title="Vehicle Routes"):
        """绘制给定路径的路线图，包括充电桩位置，并区分电动车和燃油车的客户。"""
        plt.figure(figsize=(10, 8))

        # 绘制配送中心的位置
        depot_position = np.array([self.instance.O.x, self.instance.O.y])
        plt.scatter(depot_position[0], depotThe main issue in the provided code was with the synchronization between the `battery_levels` and `visited_charging_stations` lists with the length of `electric_routes`. This desynchronization caused `IndexError` when accessing these lists during the algorithm's execution.

Here's the corrected version of your code:

1. Added the missing `sync_visited_stations()` method to ensure synchronization.
2. Updated the `copy()` method to correctly copy the `visited_charging_stations` list.
3. Added necessary checks and synchronization before manipulating `battery_levels` and `visited_charging_stations`.

With these changes, the code should correctly handle the operations without running into the previous index errors. If you encounter any further issues, please let me know!
