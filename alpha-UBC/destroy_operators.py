def random_removal(instance, solution, rnd, num_to_remove):
    # 随机移除 num_to_remove 个客户
    new_solution = solution.copy()
    customers = list(new_solution["remaining_customers"])
    rnd.shuffle(customers)
    removed_customers = customers[:num_to_remove]

    for customer in removed_customers:
        for route in new_solution["electric"]:
            if customer in route:
                route.remove(customer)
                break

    new_solution["removed_customers"].extend(removed_customers)
    return new_solution

def worst_removal(instance, solution, rnd, num_to_remove):
    customers = list(solution["remaining_customers"])
    removal_candidates = sorted(customers, key=lambda c: instance.customer_demand[c], reverse=True)
    removed_customers = removal_candidates[:num_to_remove]
    for customer in removed_customers:
        solution["removed_customers"].append(customer)
        solution["remaining_customers"].remove(customer)
    return solution

def cluster_removal(instance, solution, rnd, num_to_remove):
    customers = list(solution["remaining_customers"])
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
        solution["removed_customers"].append(customer)
        solution["remaining_customers"].remove(customer)
    return solution


