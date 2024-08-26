import random
from typing import List, Dict


class DestructionOperators:
    def __init__(self, solution):
        self.solution = solution

    def random_removal(self, current_solution, rnd_state, **kwargs):
        """
        随机移除 (Random Removal, RR): 随机选择一部分客户并将其从路径中移除。
        """
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        customers = list(self.solution.customers)
        removed_customers = random.sample(customers, num_customers_to_remove)
        for customer in removed_customers:
            self.solution.remove_customer(customer)
        return removed_customers

    def calculate_relatedness(self, customer_i: int, customer_j: int, dij: Dict[int, Dict[int, float]],
                              qi: Dict[int, float], qj: Dict[int, float], FT1i: Dict[int, float],
                              FT1j: Dict[int, float], lij: Dict[int, Dict[int, float]], phi: List[float]) -> float:
        """
        计算顾客之间的相关性，用于质量肖移除 (Quality Shaw Removal, QSR) 算子。
        """
        distance = dij[customer_i][customer_j]

        relatedness = (phi[0] * distance +
                       phi[1] * abs(qi[customer_i] - qj[customer_j]) +
                       phi[2] * abs(FT1i[customer_i] - FT1j[customer_j]) +
                       phi[3] * lij[customer_i][customer_j])

        return relatedness

    def quality_shaw_removal(self, current_solution, rnd_state, **kwargs):
        """
        质量肖移除 (Quality Shaw Removal, QSR): 根据顾客之间的相关性来移除顾客。
        """
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        phi = kwargs.get('phi')
        if phi is None:
            raise ValueError("phi 参数缺失")

        customers = list(self.solution.customers)
        reference_customer = random.choice(customers)
        relatedness_scores = {}
        for customer in customers:
            if customer != reference_customer:
                relatedness_scores[customer] = self.calculate_relatedness(
                    reference_customer, customer,
                    self.solution.distance(reference_customer, customer),
                    self.solution.demand(reference_customer),
                    self.solution.demand(customer),
                    self.solution.freshness_time(reference_customer),
                    self.solution.freshness_time(customer),
                    self.solution.same_route(reference_customer, customer),
                    phi
                )
        sorted_customers = sorted(relatedness_scores.items(), key=lambda x: x[1])
        removed_customers = [reference_customer] + [customer for customer, _ in
                                                    sorted_customers[:num_customers_to_remove - 1]]
        for customer in removed_customers:
            self.solution.remove_customer(customer)
        return removed_customers

    def worst_cost_removal(self, curr_solution, rnd_state, **kwargs):
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        # 确保 num_customers_to_remove 是整数
        num_customers_to_remove = int(num_customers_to_remove)

        removal_costs = {}

        # 将电动车和燃油车的所有客户合并
        all_routes = curr_solution.electric_routes + curr_solution.fuel_routes

        for route_idx, route_tuple in enumerate(all_routes):
            route = route_tuple[0]
            for i, customer in enumerate(route[1:-1]):  # 避免移除起点和终点
                original_cost = curr_solution.objective_value
                curr_solution.remove_customer(customer)
                new_cost = curr_solution.calculate_total_cost()
                removal_costs[(customer, route_idx, i)] = original_cost - new_cost
                curr_solution.add_customer(route_idx, i + 1, customer)  # 恢复客户到原位置

        # 按照成本增量从大到小排序，选择要移除的客户
        sorted_customers = sorted(removal_costs.items(), key=lambda x: x[1], reverse=True)
        removed_customers = [(customer, route_idx) for (customer, route_idx, _), _ in
                             sorted_customers[:num_customers_to_remove]]

        # 从解决方案中移除选择的客户
        for customer, route_idx in removed_customers:
            curr_solution.remove_customer(customer)

        return [customer for customer, _ in removed_customers]

    def worst_travel_cost_removal(self, current_solution, rnd_state, **kwargs):
        """
        最差旅行成本移除 (Worst Travel Cost Removal, WCTR): 类似于WCR，但只考虑车辆的旅行成本而非总配送成本。
        """
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        removal_costs = {}
        for customer in self.solution.customers:
            original_cost = self.solution.travel_cost()
            self.solution.remove_customer(customer)
            new_cost = self.solution.travel_cost()
            self.solution.add_customer(customer)  # 恢复客户
            removal_costs[customer] = original_cost - new_cost
        sorted_customers = sorted(removal_costs.items(), key=lambda x: x[1], reverse=True)
        removed_customers = [customer for customer, _ in sorted_customers[:num_customers_to_remove]]
        for customer in removed_customers:
            self.solution.remove_customer(customer)
        return removed_customers

    def worst_time_satisfaction_removal(self, current_solution, rnd_state, **kwargs):
        """
        最差时间满意度移除 (Worst Time Satisfaction Removal, WTSR): 根据时间满意度的升序来移除满意度最低的客户。
        """
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        time_satisfaction = {customer: self.solution.time_satisfaction(customer) for customer in
                             self.solution.customers}
        sorted_customers = sorted(time_satisfaction.items(), key=lambda x: x[1])
        removed_customers = [customer for customer, _ in sorted_customers[:num_customers_to_remove]]
        for customer in removed_customers:
            self.solution.remove_customer(customer)
        return removed_customers

    def time_satisfaction_similarity_removal(self, current_solution, rnd_state, **kwargs):
        """
        时间满意度相似移除 (Time Satisfaction Similarity Removal, TSSR): 移除与参考顾客时间满意度最相似的若干顾客。
        """
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        customers = list(self.solution.customers)
        reference_customer = random.choice(customers)
        satisfaction_gap = {}
        for customer in customers:
            if customer != reference_customer:
                satisfaction_gap[customer] = abs(
                    self.solution.time_satisfaction(customer) - self.solution.time_satisfaction(reference_customer))
        sorted_customers = sorted(satisfaction_gap.items(), key=lambda x: x[1])
        removed_customers = [reference_customer] + [customer for customer, _ in
                                                    sorted_customers[:num_customers_to_remove - 1]]
        for customer in removed_customers:
            self.solution.remove_customer(customer)
        return removed_customers

# # 示例调用
# if __name__ == "__main__":
#     file_path = "c101_21.txt"  # 替换为您的实际文件路径
#     locations_data, vehicles_data = read_solomon_instance(file_path)
#     locations = [Location(**loc) for loc in locations_data]
#     instance = CustomEVRPInstance(locations, vehicles_data)
#
#     # 创建一个假设的解决方案
#     solution = Solution(instance)
#
#     destruction_ops = DestructionOperators(solution)
#     removed_customers = destruction_ops.random_removal(num_customers_to_remove=3)
#     print(f"Removed customers: {removed_customers}")
