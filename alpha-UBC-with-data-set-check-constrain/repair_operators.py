import numpy as np
import copy

def greedy_insertion(instance, state):
    """
    Greedy insertion logic: Inserts removed customers into the best position in the routes.
    """
    for customer in state.unassigned.copy():
        best_cost = float('inf')
        best_position = None
        best_route = None

        for route_index, route in enumerate(state.routes):
            for i in range(len(route) + 1):
                if can_insert(instance, state, route, route_index, i, customer):
                    cost = calculate_insertion_cost(instance, state, route, i, customer)
                    if cost < best_cost:
                        best_cost = cost
                        best_position = i
                        best_route = route

        if best_route is not None:
            best_route.insert(best_position, customer)
            state.unassigned.remove(customer)
    return state

def random_insertion(instance, state, rnd):
    """
    Random insertion logic: Randomly inserts removed customers into the best position in the routes.
    """
    for customer in state.unassigned.copy():
        best_route = None
        best_cost = float('inf')
        best_pos = None

        for route_index, route in enumerate(state.routes):
            for pos in range(len(route) + 1):
                if can_insert(instance, state, route, route_index, pos, customer):
                    cost = calculate_insertion_cost(instance, state, route, pos, customer)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = route
                        best_pos = pos

        if best_route is not None:
            best_route.insert(best_pos, customer)
        else:
            # 如果没有找到合适的位置，将其添加到新的路径
            state.routes.append([customer])

        state.unassigned.remove(customer)

    return state

def worst_insertion(instance, state):
    """
    Worst insertion logic: Inserts removed customers into the worst position in the routes.
    """
    for customer in state.unassigned.copy():
        worst_cost = float('-inf')
        worst_position = None
        worst_route = None

        for route_index, route in enumerate(state.routes):
            for i in range(len(route) + 1):
                if can_insert(instance, state, route, route_index, i, customer):
                    cost = calculate_insertion_cost(instance, state, route, i, customer)
                    if cost > worst_cost:
                        worst_cost = cost
                        worst_position = i
                        worst_route = route

        if worst_route is not None:
            worst_route.insert(worst_position, customer)
            state.unassigned.remove(customer)

    return state

def can_insert(instance, state, route, route_index, pos, customer):
    """
    Checks if inserting customer does not exceed vehicle capacity or violate constraints.
    """
    total_load = sum(instance.customer_demand[route[i] - 1] for i in range(len(route))) + instance.customer_demand[customer - 1]
    if route_index < instance.k_e:
        return total_load <= instance.Q_e
    else:
        return total_load <= instance.Q_f

def calculate_insertion_cost(instance, state, route, idx, customer):
    """
    Computes the insertion cost for inserting customer in route_in at idx.
    """
    distances = instance.distance_matrix
    pred = state.O if idx == 0 else route[idx - 1]
    succ = state.O_prime if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return distances[pred][customer] + distances[customer][succ] - distances[pred][succ]
