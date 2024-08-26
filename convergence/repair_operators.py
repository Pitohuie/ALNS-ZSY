import random
from typing import List


class InsertionOperators:
    def __init__(self, instance):
        self.instance = instance

    def get_unassigned_customers(self, curr_solution):
        # 从实例中获取所有客户的索引
        all_customers = set(self.instance.customer_indices)
        assigned_customers = set()

        # 收集所有已分配到路线中的客户
        for route in curr_solution.electric_routes + curr_solution.fuel_routes:
            assigned_customers.update(route[0][1:-1])  # 排除路径中的起点和终点（假设为depot）

        # 找出未分配的客户
        unassigned_customers = all_customers - assigned_customers
        return list(unassigned_customers)

    def greedy_insertion(self, curr_solution, *args, **kwargs):
        print("Running greedy_insertion")
        unassigned_customers = self.get_unassigned_customers(curr_solution)
        for customer in unassigned_customers:
            best_position = self.find_best_position(curr_solution, customer)
            if best_position:
                route_index, insert_position = best_position
                curr_solution.add_customer(route_index, insert_position, customer)

        return curr_solution

    def sequence_greedy_insertion(self, curr_solution, rnd_state, **kwargs):
        customers = kwargs.get('customers', self.get_unassigned_customers(curr_solution))
        while customers:
            customer = rnd_state.choice(customers)
            best_cost = float('inf')
            best_position = None

            for route_idx, route in enumerate(curr_solution.electric_routes + curr_solution.fuel_routes):
                for position in range(1, len(route[0])):
                    cost = self.calculate_insertion_cost(route[0], customer, position)
                    if cost < best_cost:
                        best_cost = cost
                        best_position = (route_idx, position)

            if best_position:
                route_index, insert_position = best_position
                curr_solution.add_customer(route_index, insert_position, customer)

            customers.remove(customer)
        return curr_solution

    def travel_cost_greedy_insertion(self, curr_solution, rnd_state, **kwargs):
        customers = kwargs.get('customers', self.get_unassigned_customers(curr_solution))
        while customers:
            best_cost = float('inf')
            best_customer = None
            best_position = None

            for customer in customers:
                for route_idx, route in enumerate(curr_solution.electric_routes + curr_solution.fuel_routes):
                    for position in range(1, len(route[0])):
                        cost = self.calculate_travel_cost(route[0], customer, position)
                        if cost < best_cost:
                            best_cost = cost
                            best_customer = customer
                            best_position = (route_idx, position)

            if best_position:
                route_index, insert_position = best_position
                curr_solution.add_customer(route_index, insert_position, best_customer)
                customers.remove(best_customer)
        return curr_solution

    def regret_insertion(self, curr_solution, rnd_state, **kwargs):
        customers = kwargs.get('customers', self.get_unassigned_customers(curr_solution))
        while customers:
            best_regret_value = float('-inf')
            best_customer = None
            best_position = None

            for customer in customers:
                costs = []
                positions = []
                for route_idx, route in enumerate(curr_solution.electric_routes + curr_solution.fuel_routes):
                    for position in range(1, len(route[0])):
                        cost = self.calculate_insertion_cost(route[0], customer, position)
                        costs.append(cost)
                        positions.append((route_idx, position))

                if len(costs) > 1:
                    sorted_costs = sorted(zip(costs, positions))
                    regret_value = sorted_costs[1][0] - sorted_costs[0][0]

                    if regret_value > best_regret_value:
                        best_regret_value = regret_value
                        best_customer = customer
                        best_position = sorted_costs[0][1]

            if best_customer is not None and best_position is not None:
                route_index, insert_position = best_position
                curr_solution.add_customer(route_index, insert_position, best_customer)
                customers.remove(best_customer)
        return curr_solution

    def sequence_regret_insertion(self, curr_solution, rnd_state, **kwargs):
        customers = kwargs.get('customers', self.get_unassigned_customers(curr_solution))
        while customers:
            customer = rnd_state.choice(customers)
            costs = []
            positions = []
            for route_idx, route in enumerate(curr_solution.electric_routes + curr_solution.fuel_routes):
                for position in range(1, len(route[0])):
                    cost = self.calculate_insertion_cost(route[0], customer, position)
                    costs.append(cost)
                    positions.append((route_idx, position))

            if len(costs) > 1:
                sorted_costs = sorted(zip(costs, positions))
                regret_value = sorted_costs[1][0] - sorted_costs[0][0]

                best_position = sorted_costs[0][1]
                route_index, insert_position = best_position
                curr_solution.add_customer(route_index, insert_position, customer)

            customers.remove(customer)
        return curr_solution

    # 辅助函数
    def calculate_insertion_cost(self, route: List[int], customer: int, position: int) -> float:
        prev_customer = route[position - 1]
        next_customer = route[position] if position < len(route) else route[-1]

        added_distance = (self.instance.d_ij[prev_customer, customer] +
                          self.instance.d_ij[customer, next_customer] -
                          self.instance.d_ij[prev_customer, next_customer])

        service_time_cost = self.instance.locations[customer].service_time

        return added_distance + service_time_cost

    def calculate_travel_cost(self, route: List[int], customer: int, position: int) -> float:
        prev_customer = route[position - 1]
        next_customer = route[position] if position < len(route) else prev_customer

        added_travel_cost = (self.instance.d_ij[prev_customer, customer] +
                             self.instance.d_ij[customer, next_customer] -
                             self.instance.d_ij[prev_customer, next_customer])

        return added_travel_cost

    def find_best_position(self, curr_solution, customer):
        best_cost = float('inf')
        best_position = None

        for route_idx, route in enumerate(curr_solution.electric_routes + curr_solution.fuel_routes):
            for position in range(1, len(route[0])):
                cost = self.calculate_insertion_cost(route[0], customer, position)
                if cost < best_cost:
                    best_cost = cost
                    best_position = (route_idx, position)

        return best_position
