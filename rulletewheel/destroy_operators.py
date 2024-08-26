import random
import numpy as np

def random_removal(instance, solution, num_to_remove):
    new_solution = solution.copy()
    for _ in range(num_to_remove):
        route = random.choice(new_solution["conventional"] + new_solution["electric"])
        if len(route) > 2:
            to_remove = random.choice(route[1:-1])
            route.remove(to_remove)
            new_solution["remaining_customers"].append(to_remove)
    return new_solution

def worst_removal(instance, solution, num_to_remove):
    new_solution = solution.copy()
    all_customers = [(customer, route, pos)
                     for route in new_solution["conventional"] + new_solution["electric"]
                     for pos, customer in enumerate(route[1:-1])]
    all_customers.sort(key=lambda x: instance.customer_demand[x[0]], reverse=True)
    for customer, route, pos in all_customers[:num_to_remove]:
        route.remove(customer)
        new_solution["remaining_customers"].append(customer)
    return new_solution

def cluster_removal(instance, solution, num_to_remove):
    new_solution = solution.copy()
    all_customers = [customer
                     for route in new_solution["conventional"] + new_solution["electric"]
                     for customer in route[1:-1]]
    clusters = []
    while all_customers:
        cluster = [all_customers.pop(0)]
        for _ in range(num_to_remove - 1):
            nearest = min(all_customers, key=lambda x: np.linalg.norm(instance.customer_positions[cluster[-1]] - instance.customer_positions[x]))
            cluster.append(nearest)
            all_customers.remove(nearest)
        clusters.append(cluster)
    for cluster in clusters:
        for customer in cluster:
            for route in new_solution["conventional"] + new_solution["electric"]:
                if customer in route:
                    route.remove(customer)
                    new_solution["remaining_customers"].append(customer)
    return new_solution

def relatedness_removal(instance, solution, num_to_remove, relatedness_threshold):
    new_solution = solution.copy()
    remaining_customers = []
    for _ in range(num_to_remove):
        route = random.choice(new_solution["conventional"] + new_solution["electric"])
        if len(route) > 2:
            to_remove = random.choice(route[1:-1])
            related_customers = [c for c in route[1:-1] if instance.distance_matrix[to_remove, c] < relatedness_threshold]
            for customer in related_customers:
                if customer in route:
                    route.remove(customer)
                    remaining_customers.append(customer)
    new_solution["remaining_customers"].extend(remaining_customers)
    return new_solution

def time_window_removal(instance, solution, num_to_remove):
    new_solution = solution.copy()
    for _ in range(num_to_remove):
        route = random.choice(new_solution["conventional"] + new_solution["electric"])
        if len(route) > 2:
            time_window_violation_customers = [customer for customer in route[1:-1] if instance.time_window_start[customer] > instance.time_window_end[customer]]
            if time_window_violation_customers:
                to_remove = random.choice(time_window_violation_customers)
            else:
                to_remove = random.choice(route[1:-1])
            route.remove(to_remove)
            new_solution["remaining_customers"].append(to_remove)
    return new_solution
