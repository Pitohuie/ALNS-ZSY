import numpy as np

def calculate_insertion_cost(instance, route, prev_customer, customer, next_customer):
    """
    计算将新客户插入到现有路线中的成本。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含距离矩阵等数据。
    route_in (list): 当前的路线。
    prev_customer (int or str): 前一个客户的ID或索引。
    customer (int or str): 当前客户的ID或索引。
    next_customer (int or str): 下一个客户的ID或索引。

    返回:
    float: 插入新客户的成本。
    """
    # 获取客户在距离矩阵中的索引
    prev_index = instance.location_id_to_index[prev_customer] if isinstance(prev_customer, str) else prev_customer
    customer_index = instance.location_id_to_index[customer] if isinstance(customer, str) else customer
    next_index = instance.location_id_to_index[next_customer] if isinstance(next_customer, str) else next_customer

    # 计算插入成本
    cost = (instance.distance_matrix[prev_index, customer_index] +
            instance.distance_matrix[customer_index, next_index] -
            instance.distance_matrix[prev_index, next_index])

    return cost


def is_feasible_conventional(instance, route, position, customer, vehicle_capacity):
    """
    检查将客户插入到常规车辆的路线中是否可行。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含需求等数据。
    route_in (list): 当前的路线。
    position (int): 插入位置。
    customer (int): 当前客户的索引。
    vehicle_capacity (float): 车辆的最大容量。

    返回:
    bool: 如果可行返回 True，否则返回 False。
    """
    # 检查容量约束
    current_load = sum(instance.customer_demand[c] for c in route if c < instance.n)
    if current_load + instance.customer_demand[customer] > vehicle_capacity:
        return False

    # 其他可行性检查，如时间窗等，可以在这里添加
    return True

def is_feasible_electric(instance, route, position, customer, vehicle_capacity, current_battery):
    """
    检查将客户插入到电动车的路线中是否可行。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含需求、电池容量等数据。
    route_in (list): 当前的路线。
    position (int): 插入位置。
    customer (int): 当前客户的索引。
    vehicle_capacity (float): 车辆的最大容量。
    current_battery (float): 当前电池容量。

    返回:
    bool: 如果可行返回 True，否则返回 False。
    """
    # 检查容量约束
    if not is_feasible_conventional(instance, route, position, customer, vehicle_capacity):
        return False

    # 检查电池容量约束
    prev_customer = route[position - 1]
    next_customer = route[position] if position < len(route) else instance.O_prime

    # 计算额外的距离
    additional_distance = (instance.distance_matrix[prev_customer, customer] +
                           instance.distance_matrix[customer, next_customer] -
                           instance.distance_matrix[prev_customer, next_customer])

    # 检查是否满足电池容量约束
    if current_battery - additional_distance < instance.B_star * instance.soc_min:
        return False

    return True

def verify_time_windows(instance, route):
    """
    验证路线中的时间窗约束。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含时间窗等数据。
    route_in (list): 当前的路线。

    返回:
    bool: 如果时间窗约束满足返回 True，否则返回 False。
    """
    current_time = 0  # 初始化当前时间
    for i in range(1, len(route)):
        customer = route[i]
        current_time += instance.travel_time_matrix[route[i - 1], customer]  # 增加行驶时间
        if current_time > instance.time_window_end[customer]:  # 检查时间窗约束
            return False
    return True

def find_nearest_charging_station(instance, customer):
    """
    找到距离当前客户最近的充电站。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含位置等数据。
    customer (int): 当前客户的索引。

    返回:
    int: 最近充电站的索引。
    """
    charging_stations = range(instance.n, instance.n + instance.m)  # 获取所有充电站的索引
    nearest_station = min(charging_stations, key=lambda s: instance.distance_matrix[customer, s])  # 找到最近的充电站
    return nearest_station

def repair_solution(instance, route):
    """
    修复给定的解决方案，移除不满足约束的节点。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含时间窗等数据。
    route_in (list): 当前的路线。

    返回:
    list: 修复后的路线。
    """
    for i in range(1, len(route) - 1):
        if route[i] >= instance.n:  # 检查是否是充电站
            route.pop(i)  # 移除充电站
            if verify_time_windows(instance, route):  # 验证时间窗约束
                return route
        elif route[i] < instance.n:  # 检查是否是客户
            removed_customer = route.pop(i)  # 移除客户
            if verify_time_windows(instance, route):  # 验证时间窗约束
                return route
            else:
                route.insert(i, removed_customer)  # 恢复客户
    return route  # 返回修复后的路线
