import random
from typing import List

class InsertionOperators:
    def __init__(self, instance):
        self.instance = instance

    def greedy_insertion(self, current_solution: List[int], customers: List[int]) -> None:
        """
        Greedy Insertion (GI) 算子: 计算每个客户在所有可插入位置的插入成本（Δz），选择成本最小的插入点进行插入。
        """
        while customers:
            best_cost = float('inf')
            best_customer = None
            best_position = None

            for customer in customers:
                for position in range(1, len(current_solution)):
                    cost = self.calculate_insertion_cost(current_solution, customer, position)
                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_position = position

            current_solution.insert(best_position, best_customer)
            customers.remove(best_customer)

    def sequence_greedy_insertion(self, current_solution: List[int], customers: List[int]) -> None:
        """
        Sequence Greedy Insertion (SGI) 算子: 随机选择一个客户，并将其插入到成本最小的位置。
        """
        while customers:
            customer = random.choice(customers)
            best_cost = float('inf')
            best_position = None

            for position in range(1, len(current_solution)):
                cost = self.calculate_insertion_cost(current_solution, customer, position)
                if cost < best_cost:
                    best_cost = cost
                    best_position = position

            current_solution.insert(best_position, customer)
            customers.remove(customer)

    def travel_cost_greedy_insertion(self, current_solution: List[int], customers: List[int]) -> None:
        """
        Travel Cost Greedy Insertion (TCGI) 算子: 计算每个客户的旅行成本，并将其插入到成本最小的位置。
        """
        while customers:
            best_cost = float('inf')
            best_customer = None
            best_position = None

            for customer in customers:
                for position in range(1, len(current_solution)):
                    cost = self.calculate_travel_cost(current_solution, customer, position)
                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_position = position

            current_solution.insert(best_position, best_customer)
            customers.remove(best_customer)

    def regret_insertion(self, current_solution: List[int], customers: List[int]) -> None:
        """
        Regret Insertion (RI) 算子: 计算每个客户的最小插入成本和次小插入成本之间的差值（regret value），
        插入后悔值最大的客户。
        """
        while customers:
            best_regret_value = float('-inf')
            best_customer = None
            best_position = None

            for customer in customers:
                costs = []
                positions = []
                for position in range(1, len(current_solution)):
                    cost = self.calculate_insertion_cost(current_solution, customer, position)
                    costs.append(cost)
                    positions.append(position)

                if len(costs) > 1:
                    # 找到最小和次小的插入成本
                    sorted_costs = sorted(zip(costs, positions))
                    regret_value = sorted_costs[1][0] - sorted_costs[0][0]  # 计算后悔值

                    if regret_value > best_regret_value:
                        best_regret_value = regret_value
                        best_customer = customer
                        best_position = sorted_costs[0][1]

            if best_customer is not None and best_position is not None:
                current_solution.insert(best_position, best_customer)
                customers.remove(best_customer)

    def sequence_regret_insertion(self, current_solution: List[int], customers: List[int]) -> None:
        """
        Sequence Regret Insertion (SRI) 算子: 随机选择一个客户，计算该客户的后悔值，并将其插入到对应的成本最小位置。
        """
        while customers:
            customer = random.choice(customers)
            costs = []
            positions = []
            for position in range(1, len(current_solution)):
                cost = self.calculate_insertion_cost(current_solution, customer, position)
                costs.append(cost)
                positions.append(position)

            if len(costs) > 1:
                # 找到最小和次小的插入成本
                sorted_costs = sorted(zip(costs, positions))
                regret_value = sorted_costs[1][0] - sorted_costs[0][0]  # 计算后悔值

                best_position = sorted_costs[0][1]
                current_solution.insert(best_position, customer)

            customers.remove(customer)

    # 辅助函数
    def calculate_insertion_cost(self, current_solution: List[int], customer: int, position: int) -> float:
        """
        计算插入成本的方法，基于客户到当前路径的插入位置的距离和服务时间。
        """
        prev_customer = current_solution[position - 1]
        next_customer = current_solution[position] if position < len(current_solution) else prev_customer

        # 插入客户后的总距离变化量
        added_distance = (self.instance.d_ij[prev_customer, customer] +
                          self.instance.d_ij[customer, next_customer] -
                          self.instance.d_ij[prev_customer, next_customer])

        # 计算增加的服务时间成本
        service_time_cost = self.instance.locations[customer].service_time

        # 总成本
        return added_distance + service_time_cost

    def calculate_travel_cost(self, current_solution: List[int], customer: int, position: int) -> float:
        """
        计算旅行成本的方法，基于车辆在插入客户后的总距离变化。
        """
        prev_customer = current_solution[position - 1]
        next_customer = current_solution[position] if position < len(current_solution) else prev_customer

        # 计算插入客户后的总旅行成本
        added_travel_cost = (self.instance.d_ij[prev_customer, customer] +
                             self.instance.d_ij[customer, next_customer] -
                             self.instance.d_ij[prev_customer, next_customer])

        return added_travel_cost

# 示例调用
# if __name__ == "__main__":
#     file_path = "c101_21.txt"  # 替换为您的实际文件路径
#     locations_data, vehicles_data = read_solomon_instance(file_path)
#     locations = [Location(**loc) for loc in locations_data]
#     instance = CustomEVRPInstance(locations, vehicles_data)
#
#     insertion_ops = InsertionOperators(instance)
#     current_solution = [0]  # 假设当前解决方案初始化为仅包含配送中心
#     customers = [1, 2, 3, 4]  # 假设有4个客户等待插入
#
#     insertion_ops.greedy_insertion(current_solution, customers)
#     print(f"Current solution after greedy insertion: {current_solution}")
