import numpy as np
import random
import copy
from d_state import State, ALNSStateManager
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance , Location
from c_constraints import Constraints
from a_read_instance import read_solomon_instance

class ALNS:
    def __init__(self, instance: CustomEVRPInstance, constraints: Constraints, initial_solution: State):
        self.instance = instance
        self.constraints = constraints
        self.state_manager = ALNSStateManager(instance, constraints)
        self.current_state = initial_solution
        self.best_state = initial_solution.copy()
        self.temperature = 1000  # 初始温度
        self.cooling_rate = 0.995  # 冷却速率

        # 算子权重和分数
        self.destruction_weights = [1] * 5  # 假设有5个破坏算子
        self.repair_weights = [1] * 5  # 假设有5个修复算子
        self.destruction_scores = [0] * 5
        self.repair_scores = [0] * 5

    def roulette_wheel_selection(self, weights):
        total_weight = sum(weights)
        pick = random.uniform(0, total_weight)
        current = 0
        for idx, weight in enumerate(weights):
            current += weight
            if current > pick:
                return idx

    def simulated_annealing_acceptance(self, candidate_cost, current_cost):
        if candidate_cost < current_cost:
            return True
        else:
            acceptance_probability = np.exp((current_cost - candidate_cost) / self.temperature)
            return random.random() < acceptance_probability

    def run(self, iterations):
        for iteration in range(iterations):
            # 选择破坏和修复算子
            destruction_operator_index = self.roulette_wheel_selection(self.destruction_weights)
            repair_operator_index = self.roulette_wheel_selection(self.repair_weights)

            # 应用破坏和修复算子
            destroyed_state = self.state_manager.apply_destruction_operator(self.current_state, destruction_operator_index)
            repaired_state = self.state_manager.apply_repair_operator(destroyed_state, repair_operator_index)

            # 计算新的成本
            candidate_cost = repaired_state.calculate_total_cost()

            # 模拟退火决定是否接受新解
            if self.simulated_annealing_acceptance(candidate_cost, self.current_state.calculate_total_cost()):
                self.current_state = repaired_state
                # 如果新解更好，更新最优解
                if candidate_cost < self.best_state.calculate_total_cost():
                    self.best_state = copy.deepcopy(self.current_state)

            # 更新温度
            self.temperature *= self.cooling_rate

            # 更新算子权重和分数
            self.update_operator_scores(destruction_operator_index, repair_operator_index, candidate_cost)

    def update_operator_scores(self, destruction_operator_index, repair_operator_index, candidate_cost):
        improvement = max(0, self.current_state.calculate_total_cost() - candidate_cost)
        self.destruction_scores[destruction_operator_index] += improvement
        self.repair_scores[repair_operator_index] += improvement

        # 更新权重
        self.destruction_weights[destruction_operator_index] = (self.destruction_weights[destruction_operator_index] + 0.1 * self.destruction_scores[destruction_operator_index]) / 1.1
        self.repair_weights[repair_operator_index] = (self.repair_weights[repair_operator_index] + 0.1 * self.repair_scores[repair_operator_index]) / 1.1

    def get_best_solution(self):
        return self.best_state

# 主要函数
if __name__ == "__main__":
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)
    constraints = Constraints(instance)

    # 初始化初始解
    initial_solution = State(instance)  # 假设State类包含初始化逻辑

    # 运行ALNS算法
    alns = ALNS(instance, constraints, initial_solution)
    alns.run(1000)  # 假设运行1000次迭代

    # 获取最优解
    best_solution = alns.get_best_solution()
    print("Best solution found:", best_solution)
