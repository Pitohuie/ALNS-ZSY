import numpy as np
from constraints import calculate_insertion_cost  # 导入 calculate_insertion_cost 函数

def construct_initial_solution(instance, E, C):
    """
    构建初始解决方案，包括电动车和燃油车的路径。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    E (list): 电动车分配的客户列表。
    C (list): 燃油车分配的客户列表。

    返回:
    dict: 初始解决方案，包括电动车和燃油车的路径，未分配的客户和已移除的客户。
    """
    electric_routes, remaining_customers = insertion_heuristic_electric(instance, E, C)  # 使用插入启发式方法构建电动车路径
    conventional_routes = insertion_heuristic_conventional(instance, remaining_customers)  # 使用插入启发式方法构建燃油车路径
    initial_solution = {
        'electric': electric_routes,
        'conventional': conventional_routes,
        'remaining_customers': remaining_customers,
        'removed_customers': [],
        'instance': instance
    }
    return initial_solution  # 返回初始解决方案

def insertion_heuristic_electric(instance, E, C):
    """
    使用插入启发式方法构建电动车路径。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    E (list): 电动车分配的客户列表。
    C (list): 燃油车分配的客户列表。

    返回:
    tuple: 电动车路径和剩余未分配的客户。
    """
    electric_routes = [[] for _ in E]  # 初始化电动车路径
    remaining_customers = list(range(instance.n_customers))  # 获取所有客户的索引

    for i in range(len(electric_routes)):
        route = electric_routes[i]  # 获取当前电动车的路径
        while remaining_customers:
            best_cost = float('inf')  # 初始化最优成本为无穷大
            best_customer = None
            best_position = None

            for customer in remaining_customers:
                for position in range(len(route) + 1):
                    prev_customer = route[position - 1] if position > 0 else 'D0'  # 获取前一个客户
                    next_customer = route[position] if position < len(route) else 'D0'  # 获取下一个客户
                    cost = calculate_insertion_cost(instance, route, prev_customer, customer, next_customer)  # 计算插入成本

                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_position = position

            if best_customer is not None:
                route.insert(best_position, best_customer)  # 在最佳位置插入客户
                remaining_customers.remove(best_customer)  # 从剩余客户中移除已分配客户

        electric_routes[i] = route  # 更新当前电动车的路径

    return electric_routes, remaining_customers  # 返回电动车路径和剩余未分配的客户

def insertion_heuristic_conventional(instance, remaining_customers):
    """
    使用插入启发式方法构建燃油车路径。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    remaining_customers (list): 剩余未分配的客户列表。

    返回:
    list: 燃油车路径列表。
    """
    routes = []
    while remaining_customers:
        route = [instance.O['id']]  # 初始化路径，起点为配送中心
        while remaining_customers:
            min_cost = float('inf')  # 初始化最小成本为无穷大
            best_customer = None
            best_position = None
            for customer in remaining_customers:
                for i in range(1, len(route) + 1):
                    cost = calculate_insertion_cost(instance, route, i, customer)  # 计算插入成本
                    if cost < min_cost:
                        min_cost = cost
                        best_customer = customer
                        best_position = i
            if best_customer is not None:
                route.insert(best_position, best_customer)  # 在最佳位置插入客户
                remaining_customers.remove(best_customer)  # 从剩余客户中移除已分配客户
            else:
                break
        route.append(instance.O_prime)  # 在路径末尾添加终点
        routes.append(route)  # 添加路径到路径列表
    return routes  # 返回燃油车路径列表

def insert_into_conventional(instance, routes, customer):
    """
    将客户插入到燃油车的路径中。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    routes (list): 燃油车路径列表。
    customer (int): 要插入的客户索引。

    返回:
    None
    """
    best_route = None
    best_insertion_cost = float('inf')  # 初始化最佳插入成本为无穷大
    best_position = None

    for route in routes:
        for p in range(1, len(route)):
            insertion_cost = calculate_insertion_cost(instance, route, p, customer)  # 计算插入成本
            if insertion_cost < best_insertion_cost:
                best_route = route
                best_insertion_cost = insertion_cost
                best_position = p

    if best_route is not None:
        best_route.insert(best_position, customer)  # 在最佳位置插入客户
