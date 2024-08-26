import numpy as np

def greedy_insertion(instance, solution):
    """
    使用贪心插入策略将已移除的客户重新插入到解决方案中。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 当前的解决方案，包含电动车路径和已移除的客户。

    返回:
    dict: 更新后的解决方案。
    """
    # 遍历所有已移除的客户
    for customer in solution["removed_customers"]:
        best_cost = float('inf')  # 初始化最优成本为无穷大
        best_position = None
        best_route = None

        # 遍历所有电动车路径，寻找最佳插入位置
        for route in solution["electric"]:
            for i in range(1, len(route)):
                cost = calculate_insertion_cost(instance, solution, route, i, customer)  # 计算插入成本
                if cost < best_cost:
                    best_cost = cost
                    best_position = i
                    best_route = route

        # 在最佳位置插入客户
        if best_route is not None:
            best_route.insert(best_position, customer)
            solution["removed_customers"].remove(customer)  # 从已移除客户列表中移除客户

    return solution  # 返回更新后的解决方案

def random_insertion(instance, solution):
    """
    使用随机插入策略将已移除的客户重新插入到解决方案中。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 当前的解决方案，包含电动车路径和已移除的客户。

    返回:
    dict: 更新后的解决方案。
    """
    new_solution = solution.copy()  # 创建解决方案的副本

    # 遍历所有已移除的客户
    for customer in new_solution["removed_customers"]:
        best_route = None
        best_cost = float('inf')
        best_pos = None

        # 遍历所有电动车路径，寻找最佳插入位置
        for route in new_solution["electric"]:
            for pos in range(1, len(route)):
                cost = calculate_insertion_cost(instance, route, pos, customer)  # 计算插入成本
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
                    best_pos = pos

        # 在最佳位置插入客户
        if best_route is not None:
            best_route.insert(best_pos, customer)
        else:
            # 如果没有找到合适的位置，将其添加到第一个路径
            new_solution["electric"][0].insert(1, customer)

    new_solution["removed_customers"] = []  # 清空已移除客户列表
    return new_solution  # 返回更新后的解决方案

def worst_insertion(instance, solution):
    """
    使用最差插入策略将已移除的客户重新插入到解决方案中。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 当前的解决方案，包含电动车路径和已移除的客户。

    返回:
    dict: 更新后的解决方案。
    """
    # 遍历所有已移除的客户
    for customer in solution["removed_customers"]:
        worst_cost = float('-inf')  # 初始化最差成本为负无穷大
        worst_position = None
        worst_route = None

        # 遍历所有电动车路径，寻找最差插入位置
        for route in solution["electric"]:
            for i in range(1, len(route)):
                cost = calculate_insertion_cost(instance, solution, route, i, customer)  # 计算插入成本
                if cost > worst_cost:
                    worst_cost = cost
                    worst_position = i
                    worst_route = route

        # 在最差位置插入客户
        if worst_route is not None:
            worst_route.insert(worst_position, customer)
            solution["removed_customers"].remove(customer)  # 从已移除客户列表中移除客户

    return solution  # 返回更新后的解决方案
