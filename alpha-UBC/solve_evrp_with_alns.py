import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ALNS-master')))

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


def plot_solution(instance, solution):
    for route in solution["electric"]:
        x = [instance.customer_positions[node][0] if node < instance.n else instance.depot_position[0] for node in route]
        y = [instance.customer_positions[node][1] if node < instance.n else instance.depot_position[1] for node in route]
        plt.plot(x, y, marker='o')

    for route in solution["conventional"]:
        x = [instance.customer_positions[node][0] if node < instance.n else instance.depot_position[0] for node in route]
        y = [instance.customer_positions[node][1] if node < instance.n else instance.depot_position[1] for node in route]
        plt.plot(x, y, marker='x')


def solve(instance):
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

    def simple_accept(candidate, current):
        return candidate.objective() < current.objective()

    def accept(rnd, best, current, candidate):
        return simple_accept(candidate, current)

    stop = MaxIterations(1000)

    # 创建 ALNS 对象
    alns = ALNS()

    # 添加破坏和修复算子
    alns.add_destroy_operator(lambda state, rnd: EVRPState(random_removal(instance, state.solution, rnd=rnd, num_to_remove=5), instance), name="random_removal")
    alns.add_destroy_operator(lambda state, rnd: EVRPState(worst_removal(instance, state.solution, rnd=rnd, num_to_remove=5), instance), name="worst_removal")
    alns.add_destroy_operator(lambda state, rnd: EVRPState(cluster_removal(instance, state.solution, rnd=rnd, num_to_remove=5), instance), name="cluster_removal")

    alns.add_repair_operator(lambda state, rnd: EVRPState(greedy_insertion(instance, state.solution), instance), name="greedy_insertion")
    alns.add_repair_operator(lambda state, rnd: EVRPState(random_insertion(instance, state.solution), instance), name="random_insertion")
    alns.add_repair_operator(lambda state, rnd: EVRPState(worst_insertion(instance, state.solution), instance), name="worst_insertion")

    # 打印操作符数量和名称以调试
    print("Number of destroy operators:", len(alns.destroy_operators))
    print("Destroy operators:", alns.destroy_operators)
    print("Number of repair operators:", len(alns.repair_operators))
    print("Repair operators:", alns.repair_operators)

    # 调试选择过程
    try:
        for _ in range(10):  # 测试10次选择操作符
            d_idx, r_idx = select(np.random, current_state, best_state)
            print(f"Selected destroy operator index: {d_idx}, repair operator index: {r_idx}")

        # 运行 ALNS
        result = alns.iterate(current_state, select, accept, stop)
        # 输出结果
        print(f"Best solution found has objective value {result.best_state.objective()}.")
        plot_solution(instance, result.best_state.solution)
        plt.show()
    except IndexError as e:
        print(f"IndexError encountered: {e}")
        print(f"Destroy operators: {alns.destroy_operators}")
        print(f"Repair operators: {alns.repair_operators}")


if __name__ == "__main__":
    instance = CustomEVRPInstance()
    instance.print_objective()

    solve(instance)
