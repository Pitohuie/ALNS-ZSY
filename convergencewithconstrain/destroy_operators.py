import random
from typing import List, Dict

class DestructionOperators:
    def __init__(self, solution, rnd_state, num_customers_to_remove):
        self.solution = solution
        self.rnd_state = rnd_state
        self.num_customers_to_remove = num_customers_to_remove  # Store this as a class attribute

    def get_customers_from_solution(self, curr_solution):
        customers = set()
        for route in curr_solution.electric_routes + curr_solution.fuel_routes:
            customers.update(route[0][1:-1])  # 忽略起点和终点
        return list(customers)



    def random_removal(self, curr_solution, *args, **kwargs):
        customers = self.get_customers_from_solution(curr_solution)

        if len(customers) <= self.num_customers_to_remove:
            print(f"Not enough customers available for removal. Only {len(customers)} customers available.")
            return curr_solution

        # 确保每条路线至少保留一个客户
        for route in curr_solution.electric_routes + curr_solution.fuel_routes:
            if len(route[0]) - 2 <= self.num_customers_to_remove:  # 减去2是因为要保留始发地和终点站
                continue  # 跳过这条路线，确保至少保留一个客户

        removed_customers = self.rnd_state.choice(customers, self.num_customers_to_remove, replace=False)

        for customer in removed_customers:
            curr_solution.remove_customer(customer)

        print("Routes after random removal:", curr_solution.electric_routes, curr_solution.fuel_routes)

        return curr_solution

    def quality_shaw_removal(self, curr_solution, phi):
        customers = self.get_customers_from_solution(curr_solution)
        if len(customers) < self.num_customers_to_remove:
            print("Not enough customers available for removal. Skipping operation.")
            return curr_solution

        reference_customer = int(self.rnd_state.choice(customers))  # 确保是标准的 int 类型
        relatedness_scores = {}

        for customer in customers:
            customer = int(customer)  # 将每个客户转换为标准的 int 类型
            if customer != reference_customer:
                relatedness_scores[customer] = self.calculate_relatedness(
                    curr_solution, reference_customer, customer, phi)

        # 根据相关性分数排序客户
        sorted_customers = sorted(relatedness_scores.items(), key=lambda x: x[1])
        removed_customers = [reference_customer] + [customer for customer, _ in
                                                    sorted_customers[:self.num_customers_to_remove - 1]]

        for customer in removed_customers:
            curr_solution.remove_customer(customer)
        return curr_solution

    def calculate_relatedness(self, curr_solution, customer_i: int, customer_j: int, phi: List[float]) -> float:
        customer_i = int(customer_i)  # 确保是标准的 int 类型
        customer_j = int(customer_j)  # 确保是标准的 int 类型

        if not isinstance(customer_i, int) or not isinstance(customer_j, int):
            raise TypeError("customer_i and customer_j must be integers representing customer indices.")

        if customer_i < 22 or customer_j < 22:
            return float('inf')  # 忽略充电站或仓库

        # 调整客户索引以适应内部数据结构
        customer_i -= 22
        customer_j -= 22

        relatedness = (
                phi[0] * curr_solution.instance.d_ij[customer_i + 22, customer_j + 22] +
                phi[1] * abs(curr_solution.instance.q_i[customer_i] - curr_solution.instance.q_i[customer_j]) +
                phi[2] * abs(curr_solution.instance.E_i[customer_i] - curr_solution.instance.E_i[customer_j]) +
                phi[3] * abs(
            curr_solution.get_route_index(customer_i + 22) - curr_solution.get_route_index(customer_j + 22))
        )
        return relatedness

    def worst_cost_removal(self, curr_solution, *args, **kwargs):
        num_customers_to_remove = kwargs.get('num_customers_to_remove', self.num_customers_to_remove)

        customers = self.get_customers_from_solution(curr_solution)
        if len(customers) < num_customers_to_remove:
            print("Not enough customers available for removal. Skipping operation.")
            return curr_solution

        removal_costs = {}
        all_routes = curr_solution.electric_routes + curr_solution.fuel_routes

        for route_idx, route_tuple in enumerate(all_routes):
            route = route_tuple[0]
            for i, customer in enumerate(route[1:-1]):  # Avoid removing depot
                original_cost = curr_solution.objective_value
                curr_solution.remove_customer(customer)
                new_cost = curr_solution.calculate_total_cost()
                removal_costs[(customer, route_idx, i)] = original_cost - new_cost
                curr_solution.add_customer(route_idx, i + 1, customer)  # Restore customer

        sorted_customers = sorted(removal_costs.items(), key=lambda x: x[1], reverse=True)
        removed_customers = [(customer, route_idx) for (customer, route_idx, _), _ in
                             sorted_customers[:num_customers_to_remove]]

        for customer, route_idx in removed_customers:
            curr_solution.remove_customer(customer)

        return curr_solution

    def worst_travel_cost_removal(self, curr_solution, *args, **kwargs):
        customers = self.get_customers_from_solution(curr_solution)
        if len(customers) < self.num_customers_to_remove:
            print("Not enough customers available for removal. Skipping operation.")
            return curr_solution

        removal_costs = {}

        for customer in customers:
            original_cost = curr_solution.calculate_total_cost()
            curr_solution.remove_customer(customer)
            new_cost = curr_solution.calculate_total_cost()

            route_index = curr_solution.get_route_index(customer)
            if route_index == -1:
                continue  # If customer not found, skip

            position = curr_solution.get_position(customer)
            if position == -1:
                continue  # If customer not found, skip

            curr_solution.add_customer(route_index, position, customer)
            removal_costs[customer] = original_cost - new_cost

        sorted_customers = sorted(removal_costs.items(), key=lambda x: x[1], reverse=True)
        removed_customers = [customer for customer, _ in sorted_customers[:self.num_customers_to_remove]]

        for customer in removed_customers:
            curr_solution.remove_customer(customer)
        return curr_solution

    def worst_time_satisfaction_removal(self, curr_solution, *args, **kwargs):
        customers = self.get_customers_from_solution(curr_solution)
        time_satisfaction = {}

        for customer in customers:
            customer = int(customer)  # 转换为标准 int 类型

            try:
                time_satisfaction[customer] = curr_solution.instance.calculate_time_satisfaction(customer)
            except IndexError as e:
                print(e)
                continue  # 跳过无效的客户索引

        sorted_customers = sorted(time_satisfaction.items(), key=lambda x: x[1])
        removed_customers = [customer for customer, _ in sorted_customers[:self.num_customers_to_remove]]

        for customer in removed_customers:
            curr_solution.remove_customer(customer)

        return curr_solution

    def time_satisfaction_similarity_removal(self, curr_solution, rnd_state, **kwargs):
        valid_customers = [c for c in curr_solution.instance.customer_indices if 22 <= c < 122]

        if not valid_customers:
            raise ValueError("No valid customers found for removal.")

        reference_customer = rnd_state.choice(valid_customers)
        reference_adjusted = reference_customer - 22

        satisfaction_gap = {}

        # Calculate time satisfaction for the reference customer
        reference_satisfaction = curr_solution.instance.calculate_time_satisfaction(reference_adjusted)

        for customer in valid_customers:
            if customer != reference_customer:
                customer_adjusted = customer - 22
                customer_satisfaction = curr_solution.instance.calculate_time_satisfaction(customer_adjusted)

                satisfaction_gap[customer] = abs(customer_satisfaction - reference_satisfaction)

        sorted_customers = sorted(satisfaction_gap.items(), key=lambda x: x[1])
        removed_customers = [reference_customer] + [customer for customer, _ in
                                                    sorted_customers[:self.num_customers_to_remove - 1]]

        for customer in removed_customers:
            curr_solution.remove_customer(customer)

        return curr_solution
