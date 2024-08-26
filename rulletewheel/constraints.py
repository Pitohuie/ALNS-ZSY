def calculate_insertion_cost(instance, route, position, customer):
    prev_customer = route[position - 1]
    next_customer = route[position] if position < len(route) else instance.O_prime

    insertion_cost = (instance.distance_matrix[prev_customer, customer] +
                      instance.distance_matrix[customer, next_customer] -
                      instance.distance_matrix[prev_customer, next_customer])

    return insertion_cost

def is_feasible_conventional(instance, route, position, customer, vehicle_capacity):
    # 检查容量约束
    current_load = sum(instance.customer_demand[c] for c in route if c < instance.n)
    if current_load + instance.customer_demand[customer] > vehicle_capacity:
        return False

    # 其他可行性检查，如时间窗等，可以在这里添加
    return True

def is_feasible_electric(instance, route, position, customer, vehicle_capacity, current_battery):
    # 检查容量约束
    if not is_feasible_conventional(instance, route, position, customer, vehicle_capacity):
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
