def random_removal(instance, solution, rnd, num_to_remove):
    """
    随机移除指定数量的客户。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 当前的解决方案，包含电动车和燃油车的路径。
    rnd (Random): 随机数生成器实例。
    num_to_remove (int): 要移除的客户数量。

    返回:
    dict: 移除客户后的新解决方案。
    """
    new_solution = solution.copy()  # 创建解决方案的副本
    customers = list(new_solution["remaining_customers"])  # 获取剩余未分配的客户列表
    rnd.shuffle(customers)  # 随机打乱客户顺序
    removed_customers = customers[:num_to_remove]  # 选择前 num_to_remove 个客户

    for customer in removed_customers:
        for route in new_solution["electric"]:
            if customer in route:
                route.remove(customer)  # 从电动车路径中移除客户
                break

    new_solution["removed_customers"].extend(removed_customers)  # 更新已移除客户列表
    return new_solution  # 返回新的解决方案

def worst_removal(instance, solution, rnd, num_to_remove):
    """
    按照需求量移除指定数量的客户，优先移除需求量大的客户。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 当前的解决方案，包含电动车和燃油车的路径。
    rnd (Random): 随机数生成器实例。
    num_to_remove (int): 要移除的客户数量。

    返回:
    dict: 移除客户后的新解决方案。
    """
    customers = list(solution["remaining_customers"])  # 获取剩余未分配的客户列表
    removal_candidates = sorted(customers, key=lambda c: instance.customer_demand[c], reverse=True)  # 按需求量降序排序
    removed_customers = removal_candidates[:num_to_remove]  # 选择前 num_to_remove 个客户

    for customer in removed_customers:
        solution["removed_customers"].append(customer)  # 更新已移除客户列表
        solution["remaining_customers"].remove(customer)  # 从剩余客户列表中移除客户

    return solution  # 返回新的解决方案

def cluster_removal(instance, solution, rnd, num_to_remove):
    """
    按照集群移除指定数量的客户。

    参数:
    instance (CustomEVRPInstance): 问题实例，包含客户和车辆数据。
    solution (dict): 当前的解决方案，包含电动车和燃油车的路径。
    rnd (Random): 随机数生成器实例。
    num_to_remove (int): 要移除的客户数量。

    返回:
    dict: 移除客户后的新解决方案。
    """
    customers = list(solution["remaining_customers"])  # 获取剩余未分配的客户列表
    clusters = instance.clusters  # 获取客户集群
    rnd.shuffle(clusters)  # 随机打乱集群顺序
    removed_customers = []

    for cluster in clusters:
        if len(removed_customers) >= num_to_remove:
            break
        for customer in cluster:
            if customer in customers:
                removed_customers.append(customer)  # 从集群中选择客户
                if len(removed_customers) >= num_to_remove:
                    break

    for customer in removed_customers:
        solution["removed_customers"].append(customer)  # 更新已移除客户列表
        solution["remaining_customers"].remove(customer)  # 从剩余客户列表中移除客户

    return solution  # 返回新的解决方案
