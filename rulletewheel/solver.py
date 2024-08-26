from instance import CustomEVRPInstance
from clustering import clustering
import numpy as np


def solve(instance):
    print("Solving the instance...")

    # 聚类
    E, C = clustering(instance)
    print("Electric Vehicle Cluster (E):", E)
    print("Conventional Vehicle Cluster (C):", C)

    # 构造初始解
    initial_solution = construct_initial_solution(instance, E, C)
    print("Initial Solution:", initial_solution)

    # 打印初始目标函数值
    instance.print_objective()

    # 这里可以添加实际的ALNS求解过程


def construct_initial_solution(instance, E, C):
    # 构造初始解，包括传统车辆和电动汽车的路径
    conventional_routes = insertion_heuristic_conventional(instance, C)
    electric_routes, remaining_customers = insertion_heuristic_electric(instance, E, C)

    # 将剩余未服务客户添加到传统路线中
    if remaining_customers:
        for customer in remaining_customers:
            insert_into_conventional(instance, conventional_routes, customer)

    solution = {"conventional": conventional_routes, "electric": electric_routes}
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

            if best_customer is None or not is_feasible(instance, route, best_position, best_customer,
                                                        vehicle_capacity):
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

            if best_customer is None or not is_feasible_electric(instance, route, best_position, best_customer,
                                                                 vehicle_capacity, current_battery):
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


def calculate_insertion_cost(instance, route, position, customer):
    # 计算插入客户的成本
    prev_customer = route[position - 1]
    next_customer = route[position] if position < len(route) else instance.O_prime

    insertion_cost = (instance.distance_matrix[prev_customer, customer] +
                      instance.distance_matrix[customer, next_customer] -
                      instance.distance_matrix[prev_customer, next_customer])

    return insertion_cost


def is_feasible(instance, route, position, customer, vehicle_capacity):
    # 检查容量约束
    current_load = sum(instance.customer_demands[c] for c in route if c < instance.n)
    if current_load + instance.customer_demands[customer] > vehicle_capacity:
        return False

    # 其他可行性检查，如时间窗等，可以在这里添加
    return True


def is_feasible_electric(instance, route, position, customer, vehicle_capacity, current_battery):
    # 检查容量约束
    if not is_feasible(instance, route, position, customer, vehicle_capacity):
        return False

    # 检查电池容量约束
    prev_customer = route[position - 1]
    next_customer = route[position] if position < len(route) else instance.O_prime

    additional_distance = (instance.distance_matrix[prev_customer, customer] +
                           instance.distance_matrix[customer, next_customer] -
                           instance.distance_matrix[prev_customer, next_customer])

    if current_battery - additional_distance < instance.B_star * instance.soc_min:
        return False

    return True


def find_nearest_charging_station(instance, customer):
    # 找到最近的充电站
    charging_stations = range(instance.n, instance.n + instance.m)
    nearest_station = min(charging_stations, key=lambda s: instance.distance_matrix[customer, s])
    return nearest_station


def verify_time_windows(instance, route):
    # 验证路径中的时间窗约束
    current_time = 0
    for i in range(1, len(route)):
        customer = route[i]
        current_time += instance.travel_time_matrix[route[i - 1], customer]
        if current_time > instance.time_window_end[customer]:
            return False
    return True


def repair_solution(instance, route):
    # 通过移除客户和充电站修复解决方案
    for i in range(1, len(route) - 1):
        if route[i] >= instance.n:  # 移除充电站
            route.pop(i)
            if verify_time_windows(instance, route):
                return route
        elif route[i] < instance.n:  # 移除客户
            removed_customer = route.pop(i)
            if verify_time_windows(instance, route):
                return route
            else:
                route.insert(i, removed_customer)  # 插回原位置
    return route


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
