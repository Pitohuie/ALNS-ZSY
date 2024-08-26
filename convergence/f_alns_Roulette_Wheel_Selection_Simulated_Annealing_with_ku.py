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
import itertools
import matplotlib.pyplot as plt
import numpy as np





# 假设你希望在 1000 次迭代后停止
stop_criterion = MaxIterations(1000)

class Solution(State):
    def __init__(self, instance: CustomEVRPInstance, electric_routes: List[Tuple[List[int], float]],
                 fuel_routes: List[Tuple[List[int], float]]):
        self.instance = instance
        self.electric_routes = electric_routes
        self.fuel_routes = fuel_routes
        self.objective_value = self.calculate_total_cost()

    def copy(self) -> 'Solution':
        copied_electric_routes = [route[:] for route, _ in self.electric_routes]
        copied_fuel_routes = [route[:] for route, _ in self.fuel_routes]
        return Solution(self.instance, copied_electric_routes, copied_fuel_routes)

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

    def add_customer(self, route_index: int, position: int, customer: int) -> None:
        if route_index < len(self.electric_routes):
            self.electric_routes[route_index][0].insert(position, customer)
        else:
            route_index -= len(self.electric_routes)
            self.fuel_routes[route_index][0].insert(position, customer)

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
        """
        绘制给定路径的路线图，包括充电桩位置，并区分电动车和燃油车的客户。
        """
        plt.figure(figsize=(10, 8))

        # 绘制配送中心的位置
        depot_position = np.array([self.instance.O.x, self.instance.O.y])
        plt.scatter(depot_position[0], depot_position[1], c='red', label='Depot', marker='s', zorder=3)

        # 绘制电动车客户
        electric_customers_positions = np.array([(self.instance.locations[loc_id].x, self.instance.locations[loc_id].y)
                                                 for route, _ in self.electric_routes
                                                 for loc_id in route if loc_id in self.instance.type_to_indices['c']])
        if electric_customers_positions.size > 0:
            plt.scatter(electric_customers_positions[:, 0], electric_customers_positions[:, 1],
                        c='blue', label='Electric Vehicle Customers', marker='o', zorder=2)

        # 绘制燃油车客户
        fuel_customers_positions = np.array([(self.instance.locations[loc_id].x, self.instance.locations[loc_id].y)
                                             for route, _ in self.fuel_routes
                                             for loc_id in route if loc_id in self.instance.type_to_indices['c']])
        if fuel_customers_positions.size > 0:
            plt.scatter(fuel_customers_positions[:, 0], fuel_customers_positions[:, 1],
                        c='orange', label='Fuel Vehicle Customers', marker='x', zorder=2)

        # 绘制充电桩
        charging_stations_positions = np.array([(loc.x, loc.y) for loc in self.instance.locations if loc.type == 'f'])
        plt.scatter(charging_stations_positions[:, 0], charging_stations_positions[:, 1], c='green',
                    label='Charging Stations', marker='^', zorder=4)

        # 使用不同颜色绘制路径
        colors = itertools.cycle(plt.cm.get_cmap('tab20').colors)

        # 绘制电动车路线
        for route, energy_consumed in self.electric_routes:
            route_positions = [self.instance.locations[loc_id].x for loc_id in route], \
                [self.instance.locations[loc_id].y for loc_id in route]
            color = next(colors)
            plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='--', color=color,
                     label=f'Electric Route: Energy Consumed {energy_consumed:.2f}', zorder=1)

        # 绘制燃油车路线
        for route, _ in self.fuel_routes:
            route_positions = [self.instance.locations[loc_id].x for loc_id in route], \
                [self.instance.locations[loc_id].y for loc_id in route]
            color = next(colors)
            plt.plot(route_positions[0], route_positions[1], marker='x', linestyle='-', color=color,
                     label='Fuel Vehicle Routes', zorder=1)

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()

def f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance: CustomEVRPInstance, initial_solution: Solution,
                                                        max_iterations: int = 1000) -> Tuple[Solution, float]:
    random_state = RandomState(42)

    # 确保传递所有需要的参数，包括 num_customers_to_remove
    num_customers_to_remove = 1
    destruction_operators = DestructionOperators(initial_solution, random_state, num_customers_to_remove)

    # 配置 ALNS
    alns = ALNS(random_state)

    # 注册破坏算子
    alns.add_destroy_operator(destruction_operators.random_removal)
    alns.add_destroy_operator(lambda curr_solution, rnd_state: destruction_operators.quality_shaw_removal(curr_solution, phi=[1, 1, 1, 1]))
    alns.add_destroy_operator(destruction_operators.worst_cost_removal)
    alns.add_destroy_operator(destruction_operators.worst_travel_cost_removal)
    alns.add_destroy_operator(destruction_operators.worst_time_satisfaction_removal)
    alns.add_destroy_operator(destruction_operators.time_satisfaction_similarity_removal)

    # 注册修复算子
    repair_operators = InsertionOperators(instance)
    alns.add_repair_operator(repair_operators.greedy_insertion)
    alns.add_repair_operator(repair_operators.sequence_greedy_insertion)
    alns.add_repair_operator(repair_operators.travel_cost_greedy_insertion)
    alns.add_repair_operator(repair_operators.regret_insertion)
    alns.add_repair_operator(repair_operators.sequence_regret_insertion)

    # 使用轮盘赌选择策略
    scores = [10, 5, 2, 0]
    select = RouletteWheel(scores=scores, decay=0.8, num_destroy=len(alns.destroy_operators),
                           num_repair=len(alns.repair_operators))

    # 使用模拟退火作为接受准则
    schedule = SimulatedAnnealing(start_temperature=0.9, end_temperature=0.003, step=0.001)

    # 使用初始解开始迭代
    result = alns.iterate(initial_solution, select, schedule, stop_criterion)

    best_solution = result.best_state
    best_cost = best_solution.objective()

    return best_solution, best_cost


if __name__ == "__main__":
    file_path = "c101_21.txt"  # Replace with your actual file path
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 生成电动车和燃油车的初始解
    electric_routes, fuel_routes = construct_initial_solution(instance)

    # 打印初始解内容，检查是否包含客户
    print("Initial Electric Routes:", electric_routes)
    print("Initial Fuel Routes:", fuel_routes)

    initial_solution = Solution(instance, electric_routes, fuel_routes)

    # 运行启发式算法
    best_solution, best_cost = f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance, initial_solution)

    print(f"Best cost: {best_cost}")
    print(f"Best solution routes: Electric - {best_solution.electric_routes}, Fuel - {best_solution.fuel_routes}")

    # 绘制最终的路线图
    best_solution.plot_routes_with_stations("Best Vehicle Routes")