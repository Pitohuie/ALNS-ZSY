# construct_initial_solution.py

from constraints import (calculate_insertion_cost, is_feasible_conventional, is_feasible_electric,
                         find_nearest_charging_station, verify_time_windows, repair_solution)

def construct_initial_solution(instance, E, C):
    # 构造初始解，包括传统车辆和电动汽车的路径
    conventional_routes = insertion_heuristic_conventional(instance, C)
    electric_routes, remaining_customers = insertion_heuristic_electric(instance, E, C)

    # 将剩余未服务客户添加到传统路线中
    if remaining_customers:
        for customer in remaining_customers:
            insert_into_conventional(instance, conventional_routes, customer)

    solution = {"conventional": conventional_routes, "electric": electric_routes, "remaining_customers": list(remaining_customers)}
    return solution

def insertion_heuristic_conventional(instance, cluster):
    routes = []
    unserved_customers = set(cluster)

    while unserved_customers:
        route = [instance.O]  # 初始化路线，包含起点
        vehicle_capacity = instance.Q_f  # 设定车辆容量

        # 初始化路线，选择未服务节点中具有最小时间窗起点的节点作为起始节点
        best_initial_customer = min(unserved_customers, key=lambda u: instance.time_window_start[u])
        route.append(best_initial_customer)
        unserved_customers.remove(best_initial_customer)

        while unserved_customers:
            best_customer = None
            best_insertion_cost = float('inf')
            best_position = None

            # 计算每个未服务客户在当前路径中的最佳插入位置
            for u in unserved_customers:
                for p in range(1, len(route)):
                    insertion_cost = calculate_insertion_cost(instance, route, p, u)
                    if insertion_cost < best_insertion_cost:
                        best_customer = u
                        best_insertion_cost = insertion_cost
                        best_position = p

            if best_customer is None or not is_feasible_conventional(instance, route, best_position, best_customer, vehicle_capacity):
                break

            route.insert(best_position, best_customer)
            unserved_customers.remove(best_customer)

        route.append(instance.O_prime)  # 添加终点
        routes.append(route)

    return routes

def insertion_heuristic_electric(instance, electric_cluster, conventional_cluster):
    routes = []
    unserved_customers = set(electric_cluster).union(set(conventional_cluster))

    while unserved_customers:
        route = [instance.O]  # 初始化路线，包含起点
        vehicle_capacity = instance.Q_e  # 设定车辆容量
        current_battery = instance.B_star  # 初始化电池电量

        # 初始化路线，选择最小 li 的未服务节点
        best_initial_customer = min(unserved_customers, key=lambda u: instance.time_window_start[u])
        route.append(best_initial_customer)
        unserved_customers.remove(best_initial_customer)

        while unserved_customers:
            best_customer = None
            best_insertion_cost = float('inf')
            best_position = None

            for u in unserved_customers:
                for p in range(1, len(route) + 1):
                    insertion_cost = calculate_insertion_cost(instance, route, p, u)
                    if insertion_cost < best_insertion_cost:
                        best_customer = u
                        best_insertion_cost = insertion_cost
                        best_position = p

            if best_customer is None or not is_feasible_electric(instance, route, best_position, best_customer, vehicle_capacity, current_battery):
                break

            route.insert(best_position, best_customer)
            unserved_customers.remove(best_customer)
            current_battery -= instance.distance_matrix[route[best_position - 1], best_customer]

            # 检查是否需要充电
            if current_battery < instance.B_star * instance.soc_min:
                nearest_charging_station = find_nearest_charging_station(instance, route[best_position])
                route.insert(best_position + 1, nearest_charging_station)
                current_battery = instance.B_star

        # 验证时间窗约束并修复解决方案
        if not verify_time_windows(instance, route):
            route = repair_solution(instance, route)

        if route[-1] != instance.O_prime:
            route.append(instance.O_prime)  # 添加终点
        routes.append(route)

    remaining_customers = unserved_customers
    return routes, remaining_customers

def insert_into_conventional(instance, routes, customer):
    # 将未服务客户插入到传统车辆的路径中
    best_route = None
    best_insertion_cost = float('inf')
    best_position = None

    for route in routes:
        for p in range(1, len(route)):
            insertion_cost = calculate_insertion_cost(instance, route, p, customer)
            if insertion_cost < best_insertion_cost:
                best_route = route
                best_insertion_cost = insertion_cost
                best_position = p

    if best_route is not None:
        best_route.insert(best_position, customer)
