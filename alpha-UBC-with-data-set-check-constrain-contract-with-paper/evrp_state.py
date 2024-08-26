import copy

class EVRPState:
    def __init__(self, routes, instance, unassigned=None):
        self.routes = routes
        self.instance = instance
        self.unassigned = unassigned if unassigned is not None else []
        self.load = [0] * len(routes)
        self.time = [0] * len(routes)
        self.battery = [instance.B_star] * len(routes)  # 假设所有路线开始时电池都是满的
        self.visited_customers = set()
        self.charging_station_visits = {i: 0 for i in range(len(instance.locations)) if instance.locations[i]['type'] == 'f'}

        for k, route in enumerate(routes):
            current_load = 0
            current_time = 0
            current_battery = self.battery[k] if k < instance.k_e else None
            for customer in route:
                if customer >= len(instance.locations):
                    raise IndexError(f"Customer index {customer} is out of bounds for locations array.")
                if instance.locations[customer]['type'] == 'c' and (customer - 1 >= len(instance.customer_demand)):
                    raise IndexError(f"Customer index {customer - 1} is out of bounds for customer demand array.")
                self.visited_customers.add(customer)
                if instance.locations[customer]['type'] == 'c':
                    demand = instance.customer_demand[customer]
                    current_load += demand
                if instance.locations[customer]['type'] == 'f':
                    if self.charging_station_visits[customer] < 2:
                        self.charging_station_visits[customer] += 1
                        current_battery = instance.B_star  # 重置电池电量
                    else:
                        raise ValueError(f"Charging station {customer} has been visited more than twice.")
                self.load[k] = current_load
                if k < instance.k_e:
                    current_battery -= self.calculate_battery_usage(k, customer)
                    self.battery[k] = current_battery
                current_time += self.calculate_travel_time(k, customer)
                self.time[k] = current_time

    def copy(self):
        return EVRPState(copy.deepcopy(self.routes), self.instance, self.unassigned.copy())

    def objective(self):
        return sum(self.route_cost(route) for route in self.routes)

    @property
    def cost(self):
        return self.objective()

    def find_route(self, customer):
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Solution does not contain customer {customer}.")

    def route_cost(self, route):
        distances = self.instance.distance_matrix
        tour = [self.instance.O] + route + [self.instance.O_prime]
        return sum(distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))

    def update_state(self, route_index, customer, load, time, battery=None):
        self.visited_customers.add(customer)
        if self.instance.locations[customer]['type'] == 'f':
            if self.charging_station_visits[customer] < 2:
                self.charging_station_visits[customer] += 1
                battery = self.instance.B_star  # 重置电池电量
            else:
                raise ValueError(f"Charging station {customer} has been visited more than twice.")
        self.load[route_index] = load
        self.time[route_index] = time
        if battery is not None:
            self.battery[route_index] = battery

    def get_load(self, route_index):
        return self.load[route_index]

    def get_time(self, route_index):
        return self.time[route_index]

    def get_battery(self, route_index):
        return self.battery[route_index] if route_index < self.instance.k_e else None

    def add_unassigned(self, customer):
        self.unassigned.append(customer)

    def remove_unassigned(self, customer):
        self.unassigned.remove(customer)

    def is_unassigned(self, customer):
        return customer in self.unassigned

    def calculate_travel_time(self, vehicle, customer):
        if len(self.routes[vehicle]) == 0:
            prev_location = self.instance.O
        else:
            prev_location = self.routes[vehicle][-1]
        return self.instance.travel_time_matrix[prev_location][customer]

    def calculate_battery_usage(self, vehicle, customer):
        if len(self.routes[vehicle]) == 0:
            prev_location = self.instance.O
        else:
            prev_location = self.routes[vehicle][-1]
        return self.instance.L_ijk[prev_location][customer]

    def is_customer_visited(self, customer):
        return customer in self.visited_customers

    def validate_route(self, route_index):
        load = 0
        time = 0
        battery = self.battery[route_index] if route_index < self.instance.k_e else None

        for customer in self.routes[route_index]:
            if customer >= len(self.instance.locations):
                raise IndexError(f"Customer index {customer} is out of bounds for locations array.")
            if self.instance.locations[customer]['type'] == 'c' and (customer - 1 >= len(self.instance.customer_demand)):
                raise IndexError(f"Customer index {customer - 1} is out of bounds for customer demand array.")
            if self.instance.locations[customer]['type'] == 'c':
                demand = self.instance.customer_demand[customer]
                load += demand
            if load > self.instance.Q_e and route_index < self.instance.k_e:
                raise ValueError(f"Route {route_index} exceeds electric vehicle load capacity.")
            if load > self.instance.Q_f and route_index >= self.instance.k_e:
                raise ValueError(f"Route {route_index} exceeds fuel vehicle load capacity.")

            if route_index < self.instance.k_e:
                battery -= self.calculate_battery_usage(route_index, customer)
                if battery < self.instance.B_star * self.instance.soc_min:
                    raise ValueError(f"Route {route_index} violates battery constraints.")
                self.battery[route_index] = battery

            if self.instance.locations[customer]['type'] == 'f':
                if self.charging_station_visits[customer] >= 2:
                    raise ValueError(f"Charging station {customer} has been visited more than twice.")
                self.charging_station_visits[customer] += 1
                battery = self.instance.B_star  # 重置电池电量

            time += self.calculate_travel_time(route_index, customer)
            if time > self.instance.time_window_end[customer]:
                raise ValueError(f"Route {route_index} violates time window constraints.")
            self.time[route_index] = time
            self.load[route_index] = load

        return True
