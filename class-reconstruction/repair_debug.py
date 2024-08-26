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
        route = curr_solution.electric_routes[route_index][0] if route_index < len(curr_solution.electric_routes) else \
        curr_solution.fuel_routes[route_index - len(curr_solution.electric_routes)][0]
        vehicle_type = 'electric' if route_index < len(curr_solution.electric_routes) else 'fuel'

        current_location = route[position - 1]
        remaining_battery = curr_solution.remaining_battery(route)
        energy_consumption = self.calculate_energy_consumption(current_location, customer, vehicle_type)
        energy_needed_to_depot = self.calculate_energy_consumption(customer, self.instance.type_to_indices['d'][0],
                                                                   vehicle_type)

        if remaining_battery < energy_consumption + energy_needed_to_depot:
            nearest_station = self.find_nearest_charging_station(current_location)
            if nearest_station and remaining_battery >= self.calculate_energy_consumption(current_location,
                                                                                          nearest_station,
                                                                                          vehicle_type):
                route.insert(position, nearest_station)
                curr_solution.battery_levels[route_index] = self.instance.B_star
                position += 1
            else:
                # 电量不足以到达客户或充电站，路径返回配送中心
                route.append(self.instance.type_to_indices['d'][0])
                curr_solution.battery_levels[route_index] = 0  # 置零以表示路径已结束
                return False  # 表示未能成功插入客户

        route.insert(position, customer)
        curr_solution.update_battery_level(route_index, energy_consumption)
        return True  # 成功插入客户

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

    def create_new_route(self, curr_solution, start_customer):
        """创建新路径，假设是电动车路径"""
        new_route = [self.instance.type_to_indices['d'][0], start_customer]  # 从配送中心开始
        energy_consumption = self.calculate_energy_consumption(new_route[0], start_customer, 'electric')
        curr_solution.battery_levels.append(self.instance.B_star - energy_consumption)
        return ([new_route, energy_consumption], 0)  # 假设新路径的初始消耗为0

    def sequence_greedy_insertion(self, curr_solution, rnd_state, **kwargs):
        customers = kwargs.get('customers', self.get_unassigned_customers(curr_solution))
        new_routes = []

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
                if not self.manage_battery_and_insert_customer(curr_solution, route_index, customer, insert_position):
                    new_routes.append(self.create_new_route(curr_solution, customer))

            customers.remove(customer)

        curr_solution.electric_routes.extend(new_routes)
        return curr_solution

    # 其他插入方法可以类似地调整逻辑...

    # 辅助函数
    def calculate_insertion_cost(self, route: List[int], customer: int, position: int) -> float:
        prev_customer = route[position - 1]
        next_customer = route[position] if position < len(route) else route[-1]

        added_distance = (self.instance.d_ij[prev_customer, customer] +
                          self.instance.d_ij[customer, next_customer] -
                          self.instance.d_ij[prev_customer, next_customer])

        service_time_cost = self.instance.locations[customer].service_time

        return added_distance + service_time_cost

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
