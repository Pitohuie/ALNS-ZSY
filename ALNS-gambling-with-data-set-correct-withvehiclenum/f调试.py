from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from numpy.random import RandomState
from destroy_operators import DestructionOperators
from repair_operators import InsertionOperators
from a_read_instance import read_solomon_instance
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from e_nnwithchargingstation import construct_initial_solution
from alns.stop import MaxIterations

# 假设你希望在 1000 次迭代后停止
stop_criterion = MaxIterations(1000)


class Solution(State):
    def __init__(self, instance, electric_routes, fuel_routes):
        self.instance = instance
        self.electric_routes = electric_routes
        self.fuel_routes = fuel_routes
        self.objective_value = self.calculate_total_cost()

    def copy(self):
        # 深拷贝解决方案对象
        copied_electric_routes = [route[:] for route, _ in self.electric_routes]
        copied_fuel_routes = [route[:] for route, _ in self.fuel_routes]
        print(f"Copying solution: Electric Routes: {copied_electric_routes}, Fuel Routes: {copied_fuel_routes}")
        return Solution(self.instance, copied_electric_routes, copied_fuel_routes)

    def calculate_total_cost(self):
        total_cost = 0.0

        # 计算电动车路径的总成本
        for route_tuple in self.electric_routes:
            route = route_tuple[0]  # 假设route_tuple[0]是路径列表
            if isinstance(route, list) and len(route) > 1:  # 确保route是非空列表
                print(f"Calculating total cost for vehicle type: electric, Route: {route}")
                route_cost = self.instance.calculate_total_cost(route, 'electric')
                print(f"Electric route cost: {route_cost}")
                total_cost += route_cost
            else:
                raise ValueError(f"Invalid route format: {route}")

        # 计算燃油车路径的总成本
        for route_tuple in self.fuel_routes:
            route = route_tuple[0]  # 假设route_tuple[0]是路径列表
            if isinstance(route, list) and len(route) > 1:  # 确保route是非空列表
                print(f"Calculating total cost for vehicle type: fuel, Route: {route}")
                route_cost = self.instance.calculate_total_cost(route, 'fuel')
                print(f"Fuel route cost: {route_cost}")
                total_cost += route_cost
            else:
                raise ValueError(f"Invalid route format: {route}")

        return total_cost

    def remove_customer(self, customer):
        for route in self.electric_routes + self.fuel_routes:
            if customer in route[0]:
                print(f"Removing customer {customer} from route: {route[0]}")
                route[0].remove(customer)
                break

    def add_customer(self, route_index, position, customer):
        if route_index < len(self.electric_routes):
            print(f"Adding customer {customer} to electric route {route_index} at position {position}")
            self.electric_routes[route_index][0].insert(position, customer)
        else:
            route_index -= len(self.electric_routes)
            print(f"Adding customer {customer} to fuel route {route_index} at position {position}")
            self.fuel_routes[route_index][0].insert(position, customer)

    def objective(self):
        # 返回目标函数值
        return self.objective_value


def f_alns_Roulette_Wheel_Selection_Simulated_Annealing(instance, initial_solution, max_iterations=1000):
    random_state = RandomState(42)

    # 破坏算子和修复算子实例化
    destruction_operators = DestructionOperators(initial_solution)
    repair_operators = InsertionOperators(instance)  # 传递 instance 参数

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
    alns.add_repair_operator(repair_operators.greedy_insertion)
    alns.add_repair_operator(repair_operators.sequence_greedy_insertion)
    alns.add_repair_operator(repair_operators.travel_cost_greedy_insertion)
    alns.add_repair_operator(repair_operators.regret_insertion)
    alns.add_repair_operator(repair_operators.sequence_regret_insertion)

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
                          num_customers_to_remove=num_customers_to_remove)

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
