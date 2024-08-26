import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加 ALNS 库的路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ALNS-master')))

# 导入自定义的 EVRP 实例和约束、破坏和修复算子
from evrp_instance import CustomEVRPInstance
from constraints import (calculate_insertion_cost, is_feasible_conventional, is_feasible_electric,
                         find_nearest_charging_station, verify_time_windows, repair_solution)
from destroy_operators import (random_removal, worst_removal, cluster_removal)
from repair_operators import (greedy_insertion, random_insertion, worst_insertion)
from clustering import clustering
from construct_initial_solution import construct_initial_solution
from alns.select.AlphaUCB import AlphaUCB
from evrp_state import EVRPState
from alns import ALNS
from max_iterations import MaxIterations
from read_instance import read_solomon_instance


def plot_solution(instance, solution):
    """
    绘制解决方案中的路径。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 解决方案，包含电动车和燃油车的路径。
    """
    for route in solution["electric"]:
        x = [instance.customer_positions[node][0] if node < instance.n else instance.depot_position[0] for node in
             route]
        y = [instance.customer_positions[node][1] if node < instance.n else instance.depot_position[1] for node in
             route]
        plt.plot(x, y, marker='o')

    for route in solution["conventional"]:
        x = [instance.customer_positions[node][0] if node < instance.n else instance.depot_position[0] for node in
             route]
        y = [instance.customer_positions[node][1] if node < instance.n else instance.depot_position[1] for node in
             route]
        plt.plot(x, y, marker='x')


def solve(instance):
    """
    解决 EVRP 问题的主要函数。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    """
    print("Solving the instance...")

    # 聚类
    E, C = clustering(instance)
    print("Electric Vehicle Cluster (E):", E)
    print("Conventional Vehicle Cluster (C):", C)

    # 构造初始解
    initial_solution = construct_initial_solution(instance, E, C)
    print("Initial Solution:", initial_solution)

    # 打印初始目标函数值
    instance.print_objective()

    # 创建初始状态
    current_state = EVRPState(initial_solution, instance)
    best_state = current_state.copy()

    # 定义选择和接受准则
    select = AlphaUCB(scores=[5, 2, 1, 0.5], alpha=0.8, num_destroy=3, num_repair=3)

    def accept(candidate, current, best, rnd):
        """
        接受准则，用于判断是否接受新解。

        参数:
        candidate (EVRPState): 候选解。
        current (EVRPState): 当前解。
        best (EVRPState): 当前最优解。
        rnd (RandomState): 随机数生成器实例。

        返回:
        bool: 如果接受新解，返回 True，否则返回 False。
        """
        try:
            # 打印类型调试信息
            print(f"candidate type: {type(candidate)}, current type: {type(current)}, best type: {type(best)}")

            # 确保候选解是 EVRPState 对象
            if isinstance(candidate, EVRPState) and isinstance(current, EVRPState):
                return candidate.objective() < current.objective()
            else:
                print("Invalid type for candidate or current in accept function")
                return False
        except Exception as e:
            print(f"An error occurred in accept function: {e}")
            return False

    stop = MaxIterations(1000)  # 定义停止准则，最多迭代 1000 次

    # 创建 ALNS 对象
    alns = ALNS()

    # 添加破坏算子
    alns.add_destroy_operator(
        lambda state, rnd: EVRPState(random_removal(instance, state.solution, rnd, num_to_remove=5), instance),
        name="random_removal")
    alns.add_destroy_operator(
        lambda state, rnd: EVRPState(worst_removal(instance, state.solution, rnd, num_to_remove=5), instance),
        name="worst_removal")
    alns.add_destroy_operator(
        lambda state, rnd: EVRPState(cluster_removal(instance, state.solution, rnd, num_to_remove=5), instance),
        name="cluster_removal")

    # 添加修复算子
    alns.add_repair_operator(
        lambda state, rnd: EVRPState(greedy_insertion(instance, state.solution), instance), name="greedy_insertion")
    alns.add_repair_operator(
        lambda state, rnd: EVRPState(random_insertion(instance, state.solution), instance), name="random_insertion")
    alns.add_repair_operator(
        lambda state, rnd: EVRPState(worst_insertion(instance, state.solution), instance), name="worst_insertion")

    try:
        # 运行 ALNS
        result = alns.iterate(current_state, select, accept, stop)

        # 输出结果
        print(f"Best solution found has objective value {result.best_state.objective()}.")
        plot_solution(instance, result.best_state.solution)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = 'D:\\2024\\ZSY-ALNS\\pythonProject1\\evrptw_instances\\c101_21.txt'  # 替换为实际路径
    locations, vehicles = read_solomon_instance(file_path)

    print("Parsed locations:", locations)  # 调试信息
    print("Parsed vehicles:", vehicles)  # 调试信息

    instance = CustomEVRPInstance(locations, vehicles)
    instance.print_objective()

    solve(instance)
