import random
from typing import List


class InsertionOperators:
    def __init__(self, instance):
        self.instance = instance

    def greedy_insertion(self, current_solution: List[int], customers: List[int], rnd_state, **kwargs) -> None:
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

    def sequence_greedy_insertion(self, current_solution: List[int], customers: List[int], rnd_state, **kwargs) -> None:
        while customers:
            customer = rnd_state.choice(customers)
            best_cost = float('inf')
            best_position = None

            for position in range(1, len(current_solution)):
                cost = self.calculate_insertion_cost(current_solution, customer, position)
                if cost < best_cost:
                    best_cost = cost
                    best_position = position

            current_solution.insert(best_position, customer)
            customers.remove(customer)

    def travel_cost_greedy_insertion(self, current_solution: List[int], customers: List[int], rnd_state, **kwargs) -> None:
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

    def regret_insertion(self, current_solution: List[int], customers: List[int], rnd_state, **kwargs) -> None:
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
                    sorted_costs = sorted(zip(costs, positions))
                    regret_value = sorted_costs[1][0] - sorted_costs[0][0]

                    if regret_value > best_regret_value:
                        best_regret_value = regret_value
                        best_customer = customer
                        best_position = sorted_costs[0][1]

            if best_customer is not None and best_position is not None:
                current_solution.insert(best_position, best_customer)
                customers.remove(best_customer)

    def sequence_regret_insertion(self, current_solution: List[int], customers: List[int], rnd_state, **kwargs) -> None:
        print(f"Received rnd_state: {rnd_state}")
        print(f"Received kwargs: {kwargs}")
        # 方法体...
        while customers:
            customer = rnd_state.choice(customers)
            costs = []
            positions = []
            for position in range(1, len(current_solution)):
                cost = self.calculate_insertion_cost(current_solution, customer, position)
                costs.append(cost)
                positions.append(position)

            if len(costs) > 1:
                sorted_costs = sorted(zip(costs, positions))
                regret_value = sorted_costs[1][0] - sorted_costs[0][0]

                best_position = sorted_costs[0][1]
                current_solution.insert(best_position, customer)

            customers.remove(customer)

    # 辅助函数
    def calculate_insertion_cost(self, current_solution: List[int], customer: int, position: int) -> float:
        prev_customer = current_solution[position - 1]
        next_customer = current_solution[position] if position < len(current_solution) else prev_customer

        added_distance = (self.instance.d_ij[prev_customer, customer] +
                          self.instance.d_ij[customer, next_customer] -
                          self.instance.d_ij[prev_customer, next_customer])

        service_time_cost = self.instance.locations[customer].service_time

        return added_distance + service_time_cost

    def calculate_travel_cost(self, current_solution: List[int], customer: int, position: int) -> float:
        prev_customer = current_solution[position - 1]
        next_customer = current_solution[position] if position < len(current_solution) else prev_customer

        added_travel_cost = (self.instance.d_ij[prev_customer, customer] +
                             self.instance.d_ij[customer, next_customer] -
                             self.instance.d_ij[prev_customer, next_customer])

        return added_travel_cost
