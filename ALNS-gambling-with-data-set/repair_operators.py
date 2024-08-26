import numpy as np

def greedy_insertion(instance, solution):
    # Greedy insertion logic
    for customer in solution["removed_customers"]:
        best_cost = float('inf')
        best_position = None
        best_route = None
        for route in solution["electric"]:
            for i in range(1, len(route)):
                cost = calculate_insertion_cost(instance, solution, route, i, customer)
                if cost < best_cost:
                    best_cost = cost
                    best_position = i
                    best_route = route
        if best_route is not None:
            best_route.insert(best_position, customer)
            solution["removed_customers"].remove(customer)
    return solution

def random_insertion(instance, solution):
    new_solution = solution.copy()

    for customer in new_solution["removed_customers"]:
        best_route = None
        best_cost = float('inf')
        best_pos = None

        for route in new_solution["electric"]:
            for pos in range(1, len(route)):
                cost = calculate_insertion_cost(instance, route, pos, customer)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
                    best_pos = pos

        if best_route is not None:
            best_route.insert(best_pos, customer)
        else:
            # 如果没有找到合适的位置，将其添加到第一个路径
            new_solution["electric"][0].insert(1, customer)

    new_solution["removed_customers"] = []
    return new_solution

def worst_insertion(instance, solution):
    # Worst insertion logic
    for customer in solution["removed_customers"]:
        worst_cost = float('-inf')
        worst_position = None
        worst_route = None
        for route in solution["electric"]:
            for i in range(1, len(route)):
                cost = calculate_insertion_cost(instance, solution, route, i, customer)
                if cost > worst_cost:
                    worst_cost = cost
                    worst_position = i
                    worst_route = route
        if worst_route is not None:
            worst_route.insert(worst_position, customer)
            solution["removed_customers"].remove(customer)
    return solution
