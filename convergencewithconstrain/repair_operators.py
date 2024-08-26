import random
from typing import List


class InsertionOperators:
    def __init__(self, instance):
        self.instance = instance

    def get_unassigned_customers(self, curr_solution):
        all_customers = set(self.instance.customer_indices)
        assigned_customers = set()

        for route in curr_solution.electric_routes + curr_solution.fuel_routes:
            assigned_customers.update(route[0][1:-1])

        unassigned_customers = all_customers - assigned_customers
        return list(unassigned_customers)

    def manage_battery_and_insert_customer(self, curr_solution, route_index, customer, position):
        is_electric_route = route_index < len(curr_solution.electric_routes)

        if is_electric_route:
            route = curr_solution.electric_routes[route_index][0]
        else:
            route = curr_solution.fuel_routes[route_index - len(curr_solution.electric_routes)][0]

        vehicle_type = 'electric' if is_electric_route else 'fuel'

        current_location = route[position - 1]  # 确保 position > 0
        adjusted_customer_index = customer - 22  # 根据之前的信息，确保这个调整是正确的

        # 检查 adjusted_customer_index 是否超出距离矩阵的边界
        if adjusted_customer_index >= len(self.instance.d_ij):
            raise IndexError(f"Customer index {adjusted_customer_index} out of bounds for distance matrix.")

        # 获取剩余电量，只对电动车路线进行检查
        remaining_battery = curr_solution.remaining_battery(route_index) if is_electric_route else None

        # 计算能耗
        energy_consumption = self.calculate_energy_consumption(current_location, customer, vehicle_type)
        energy_needed_to_depot = self.calculate_energy_consumption(customer, self.instance.type_to_indices['d'][0],
                                                                   vehicle_type)

        # 根据剩余电量和能耗的关系，决定是否可以插入客户
        if is_electric_route and remaining_battery is not None:
            if remaining_battery < energy_consumption + energy_needed_to_depot:
                # 如果剩余电量不足以到达客户和返回配送中心，则需要采取行动
                # 例如，你可以将车返回到充电桩或者配送中心
                curr_solution.return_to_depot(route_index)
            else:
                # 插入客户
                curr_solution.update_battery_level(route_index, energy_consumption)
                route.insert(position, customer)
        else:
            # 燃油车直接插入客户
            route.insert(position, customer)

    def greedy_insertion(self, curr_solution, *args, **kwargs):
        print("Running greedy_insertion")
        unassigned_customers = self.get_unassigned_customers(curr_solution)
        new_routes = []

        for customer in unassigned_customers:
            best_position = self.find_best_position(curr_solution, customer)
            if best_position:
                route_index, insert_position = best_position
                if not self.manage_battery_and_insert_customer(curr_solution, route_index, customer, insert_position):
                    # 如果无法插入客户，生成新路径
                    new_routes.append(self.create_new_route(curr_solution, customer))

        curr_solution.electric_routes.extend(new_routes)
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
                self.manage_battery_and_insert_customer(curr_solution, route_index, customer, insert_position)

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
                self.manage_battery_and_insert_customer(curr_solution, route_index, best_customer, insert_position)
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
                self.manage_battery_and_insert_customer(curr_solution, route_index, best_customer, insert_position)
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
                self.manage_battery_and_insert_customer(curr_solution, route_index, customer, insert_position)

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

    def calculate_energy_consumption(self, current_location, next_location, vehicle_type):
        distance = self.instance.d_ij[current_location, next_location]
        if vehicle_type == 'electric':
            return distance * self.instance.energy_consumption_rate_electric
        else:
            return distance * self.instance.energy_consumption_rate_fuel

    def find_nearest_charging_station(self, current_location):
        charging_stations = self.instance.type_to_indices['f']
        nearest_station = min(charging_stations, key=lambda station: self.instance.d_ij[current_location, station],
                              default=None)
        return nearest_station

    def create_new_route(self, curr_solution, customer):
        """创建一个新路线，从配送中心出发，访问客户，然后返回配送中心。"""
        new_route = [self.instance.depot_index, customer, self.instance.depot_index]
        return (new_route, 0)  # 这里的 0 表示初始能量消耗为 0
