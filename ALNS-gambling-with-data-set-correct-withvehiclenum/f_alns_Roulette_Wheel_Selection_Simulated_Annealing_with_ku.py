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
        # 深拷贝解决方案对象
        copied_electric_routes = [route[:] for route, _ in self.electric_routes]
        copied_fuel_routes = [route[:] for route, _ in self.fuel_routes]
        return Solution(self.instance, copied_electric_routes, copied_fuel_routes)

    def calculate_total_cost(self) -> float:
        total_cost = 0.0

        # 计算电动车路径的总成本
        for route_tuple in self.electric_routes:
            route = route_tuple[0]  # 假设route_tuple[0]是路径列表
            if isinstance(route, list) and len(route) > 1:  # 确保route是非空列表
                route_cost = self.instance.calculate_total_cost(route, 'electric')
                total_cost += route_cost

        # 计算燃油车路径的总成本
        for route_tuple in self.fuel_routes:
            route = route_tuple[0]  # 假设route_tuple[0]是路径列表
            if isinstance(route, list) and len(route) > 1:  # 确保route是非空列表
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
        # 返回目标函数值
        return self.objective_value

def f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance: CustomEVRPInstance, initial_solution: Solution,
                                                        max_iterations: int = 1000) -> Tuple[Solution, float]:
    random_state = RandomState(42)

    # 破坏算子和修复算子实例化
    destruction_operators = DestructionOperators(initial_solution)
    repair_operators = InsertionOperators(instance)

    # 配置 ALNS
    alns = ALNS(random_state)

    # 注册破坏算子
    alns.add_destroy_operator(destruction_operators.random_removal)
    alns.add_destroy_operator(destruction_operators.quality_shaw_removal)
    alns.add_destroy_operator(destruction_operators.worst_cost_removal)
    alns.add_destroy_operator(destruction_operators.worst_travel_cost_removal)
    alns.add_destroy_operator(destruction_operators.worst_time_satisfaction_removal)
    alns.add_destroy_operator(destruction_operators.time_satisfaction_similarity_removal)

    # 注册修复算子
    alns.add_repair_operator(lambda sol, rnd_state, **kwargs: repair_operators.greedy_insertion(sol, rnd_state=rnd_state, **kwargs))
    alns.add_repair_operator(lambda sol, rnd_state, **kwargs: repair_operators.sequence_greedy_insertion(sol, rnd_state=rnd_state, **kwargs))
    alns.add_repair_operator(lambda sol, rnd_state, **kwargs: repair_operators.travel_cost_greedy_insertion(sol, rnd_state=rnd_state, **kwargs))
    alns.add_repair_operator(lambda sol, rnd_state, **kwargs: repair_operators.regret_insertion(sol, rnd_state=rnd_state, **kwargs))
    alns.add_repair_operator(lambda sol, rnd_state, **kwargs: repair_operators.sequence_regret_insertion(sol, rnd_state=rnd_state, **kwargs))

    # 初始化轮盘赌选择策略的初始分数
    initial_scores_destroy = [1] * len(alns.destroy_operators)  # 对于每个破坏算子，初始分数为1
    initial_scores_repair = [1] * len(alns.repair_operators)    # 对于每个修复算子，初始分数为1

    # 使用轮盘赌选择 (Roulette Wheel) 策略
    scores = [10, 5, 2, 0]  # 全部为非负数
    select = RouletteWheel(scores=scores, decay=0.8, num_destroy=len(alns.destroy_operators),
                           num_repair=len(alns.repair_operators))

    # 使用模拟退火 (Simulated Annealing) 作为接受准则
    schedule = SimulatedAnnealing(start_temperature=0.9, end_temperature=0.003, step=0.001)

    # 确保定义 num_customers_to_remove
    num_customers_to_remove = 5  # 你可以调整这个值

    # 使用初始解开始迭代
    result = alns.iterate(initial_solution, select, schedule, stop_criterion,
                          num_customers_to_remove=num_customers_to_remove, rnd_state=random_state)

    # 最终结果
    best_solution = result.best_state
    best_cost = best_solution.objective()

    return best_solution, best_cost


# 示例调用
if __name__ == "__main__":
    file_path = "c101_21.txt"  # 替换为实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 生成电动车和燃油车的初始解
    electric_routes, fuel_routes = construct_initial_solution(instance)

    # 创建初始解
    initial_solution = Solution(instance, electric_routes, fuel_routes)

    # 调用 ALNS 函数
    best_solution, best_cost = f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance, initial_solution)
    print(f"Best cost: {best_cost}")
    print(f"Best solution routes: Electric - {best_solution.electric_routes}, Fuel - {best_solution.fuel_routes}")
