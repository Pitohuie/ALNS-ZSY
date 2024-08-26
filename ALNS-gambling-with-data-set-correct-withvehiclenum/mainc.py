from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from numpy.random import RandomState

from destroy_operators import DestructionOperators
from repair_operators import InsertionOperators
from a_read_instance import read_solomon_instance
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from c_constraints import Constraints
from e_initial_solution_with_clustering import clustering
from e_nnwithchargingstation import construct_initial_solution


class Solution(State):
    def __init__(self, instance, routes, constraints):
        self.instance = instance
        self.routes = routes
        self.constraints = constraints
        self.objective = self.calculate_total_cost()

    def copy(self):
        # 深拷贝解决方案对象
        return Solution(self.instance, [route[:] for route in self.routes], self.constraints)

    def calculate_total_cost(self):
        # 根据实际情况计算解决方案的总成本
        total_cost = 0.0
        for route in self.routes:
            total_cost += self.instance.calculate_total_cost(route, 'electric')
        return total_cost

    def remove_customer(self, customer):
        # 从路径中移除客户
        for route in self.routes:
            if customer in route:
                route.remove(customer)
                break

    def add_customer(self, route_index, position, customer):
        # 在路径中添加客户
        self.routes[route_index].insert(position, customer)

    def objective(self):
        # 返回目标函数值
        return self.calculate_total_cost()


def f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance, initial_solution, constraints, max_iterations=1000):
    random_state = RandomState(42)
    destruction_operators = DestructionOperators()
    repair_operators = InsertionOperators()
    alns = ALNS(random_state)

    # 注册破坏算子
    alns.add_destroy_operator(destruction_operators.random_removal)
    alns.add_destroy_operator(destruction_operators.quality_shaw_removal)
    alns.add_destroy_operator(destruction_operators.worst_cost_removal)
    alns.add_destroy_operator(destruction_operators.worst_travel_cost_removal)
    alns.add_destroy_operator(destruction_operators.worst_time_satisfaction_removal)
    alns.add_destroy_operator(destruction_operators.time_satisfaction_similarity_removal)

    # 注册修复算子
    alns.add_repair_operator(repair_operators.greedy_insertion)
    alns.add_repair_operator(repair_operators.sequence_greedy_insertion)
    alns.add_repair_operator(repair_operators.travel_cost_greedy_insertion)
    alns.add_repair_operator(repair_operators.regret_insertion)
    alns.add_repair_operator(repair_operators.sequence_regret_insertion)

    # 使用轮盘赌选择 (Roulette Wheel) 策略
    select = RouletteWheel(random_state)

    # 使用模拟退火 (Simulated Annealing) 作为接受准则
    schedule = SimulatedAnnealing(0.9, 0.003, random_state)

    # 使用初始解开始迭代
    result = alns.iterate(initial_solution, select, schedule, max_iterations)

    # 最终结果
    best_solution = result.best_state
    best_cost = best_solution.objective()

    return best_solution, best_cost

if __name__ == "__main__":
    file_path = "c101_21.txt"  # 替换为实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)
    constraints = Constraints(instance)

    # 构造初始解
    electric_routes, fuel_routes = construct_initial_solution(instance)

    # 初始化Solution对象
    initial_solution = Solution(instance, electric_routes + fuel_routes, constraints)

    # 调用 ALNS 函数
    best_solution, best_cost = f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance, initial_solution, constraints)

    print(f"Best cost: {best_cost}")
    print(f"Best solution routes: {best_solution.routes}")
