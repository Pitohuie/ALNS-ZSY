import random
import copy

def random_removal(instance, state, rnd, num_to_remove):
    """
    随机移除 num_to_remove 个客户
    """
    new_state = state.copy()
    customers = list(new_state.unassigned)
    rnd.shuffle(customers)
    removed_customers = customers[:num_to_remove]

    for customer in removed_customers:
        for route in new_state.routes:
            if customer in route:
                route.remove(customer)
                break

    new_state.unassigned.extend(removed_customers)
    return remove_empty_routes(new_state)

def worst_removal(instance, state, rnd, num_to_remove):
    """
    根据客户需求量从大到小移除 num_to_remove 个客户
    """
    new_state = state.copy()
    customers = list(new_state.unassigned)
    removal_candidates = sorted(customers, key=lambda c: instance.customer_demand[c - 1], reverse=True)
    removed_customers = removal_candidates[:num_to_remove]

    for customer in removed_customers:
        new_state.unassigned.append(customer)
        for route in new_state.routes:
            if customer in route:
                route.remove(customer)
                break

    return remove_empty_routes(new_state)

def cluster_removal(instance, state, rnd, num_to_remove):
    """
    根据聚类移除 num_to_remove 个客户
    """
    new_state = state.copy()
    customers = list(new_state.unassigned)
    clusters = instance.clusters
    rnd.shuffle(clusters)
    removed_customers = []

    for cluster in clusters:
        if len(removed_customers) >= num_to_remove:
            break
        for customer in cluster:
            if customer in customers:
                removed_customers.append(customer)
                if len(removed_customers) >= num_to_remove:
                    break

    for customer in removed_customers:
        new_state.unassigned.append(customer)
        for route in new_state.routes:
            if customer in route:
                route.remove(customer)
                break

    return remove_empty_routes(new_state)

def remove_empty_routes(state):
    """
    移除空的路径
    """
    state.routes = [route for route in state.routes if route]
    return state

# 示例用法
file_path = 'path_to_your_dataset_file.txt'
locations, vehicles = read_solomon_instance(file_path)
instance = CustomEVRPInstance(locations, vehicles)
initial_routes = [[1, 2, 3], [4, 5, 6]]  # 示例初始路径
state = EVRPState(initial_routes, instance)

rnd = random.Random(42)
num_to_remove = customers_to_remove(instance)

new_state = random_removal(instance, state, rnd, num_to_remove)
print(new_state.routes)
print(new_state.unassigned)
