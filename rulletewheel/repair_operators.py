import random
from constraints import calculate_insertion_cost, is_feasible_conventional, is_feasible_electric

def greedy_insertion(instance, solution):
    new_solution = solution.copy()
    while new_solution["remaining_customers"]:
        customer = new_solution["remaining_customers"].pop(0)
        best_route, best_pos, best_cost = None, None, float('inf')
        for route in new_solution["conventional"] + new_solution["electric"]:
            for pos in range(1, len(route)):
                cost = calculate_insertion_cost(instance, route, pos, customer)
                if cost < best_cost:
                    if route in new_solution["conventional"]:
                        feasible = is_feasible_conventional(instance, route, pos, customer, instance.Q_f)
                    else:
                        feasible = is_feasible_electric(instance, route, pos, customer, instance.Q_e, instance.B_star)
                    if feasible:
                        best_route, best_pos, best_cost = route, pos, cost
        if best_route is not None:
            best_route.insert(best_pos, customer)
    return new_solution

def regret_insertion(instance, solution):
    new_solution = solution.copy()
    while new_solution["remaining_customers"]:
        best_customer, best_route, best_pos, best_regret = None, None, None, float('-inf')
        for customer in new_solution["remaining_customers"]:
            best_cost, second_best_cost = float('inf'), float('inf')
            for route in new_solution["conventional"] + new_solution["electric"]:
                for pos in range(1, len(route)):
                    cost = calculate_insertion_cost(instance, route, pos, customer)
                    if cost < best_cost:
                        if route in new_solution["conventional"]:
                            feasible = is_feasible_conventional(instance, route, pos, customer, instance.Q_f)
                        else:
                            feasible = is_feasible_electric(instance, route, pos, customer, instance.Q_e, instance.B_star)
                        if feasible:
                            second_best_cost = best_cost
                            best_cost = cost
                    elif cost < second_best_cost:
                        if route in new_solution["conventional"]:
                            feasible = is_feasible_conventional(instance, route, pos, customer, instance.Q_f)
                        else:
                            feasible = is_feasible_electric(instance, route, pos, customer, instance.Q_e, instance.B_star)
                        if feasible:
                            second_best_cost = cost
            regret = second_best_cost - best_cost
            if regret > best_regret:
                best_customer, best_route, best_pos, best_regret = customer, route, pos, regret
        new_solution["remaining_customers"].remove(best_customer)
        best_route.insert(best_pos, best_customer)
    return new_solution

def random_insertion(instance, solution):
    new_solution = solution.copy()
    while new_solution["remaining_customers"]:
        customer = new_solution["remaining_customers"].pop(0)
        route = random.choice(new_solution["conventional"] + new_solution["electric"])
        pos = random.randint(1, len(route) - 1)
        if route in new_solution["conventional"]:
            feasible = is_feasible_conventional(instance, route, pos, customer, instance.Q_f)
        else:
            feasible = is_feasible_electric(instance, route, pos, customer, instance.Q_e, instance.B_star)
        if feasible:
            route.insert(pos, customer)
    return new_solution

def worst_insertion(instance, solution):
    new_solution = solution.copy()
    while new_solution["remaining_customers"]:
        customer = new_solution["remaining_customers"].pop(0)
        worst_route, worst_pos, worst_cost = None, None, float('-inf')
        for route in new_solution["conventional"] + new_solution["electric"]:
            for pos in range(1, len(route)):
                cost = calculate_insertion_cost(instance, route, pos, customer)
                if cost > worst_cost:
                    if route in new_solution["conventional"]:
                        feasible = is_feasible_conventional(instance, route, pos, customer, instance.Q_f)
                    else:
                        feasible = is_feasible_electric(instance, route, pos, customer, instance.Q_e, instance.B_star)
                    if feasible:
                        worst_route, worst_pos, worst_cost = route, pos, cost
        if worst_route is not None:
            worst_route.insert(worst_pos, customer)
    return new_solution

def farthest_insertion(instance, solution):
    new_solution = solution.copy()
    while new_solution["remaining_customers"]:
        customer = new_solution["remaining_customers"].pop(0)
        farthest_route, farthest_pos, farthest_cost = None, None, float('-inf')
        for route in new_solution["conventional"] + new_solution["electric"]:
            for pos in range(1, len(route)):
                cost = calculate_insertion_cost(instance, route, pos, customer)
                if cost > farthest_cost:
                    if route in new_solution["conventional"]:
                        feasible = is_feasible_conventional(instance, route, pos, customer, instance.Q_f)
                    else:
                        feasible = is_feasible_electric(instance, route, pos, customer, instance.Q_e, instance.B_star)
                    if feasible:
                        farthest_route, farthest_pos, farthest_cost = route, pos, cost
        if farthest_route is not None:
            farthest_route.insert(farthest_pos, customer)
    return new_solution
