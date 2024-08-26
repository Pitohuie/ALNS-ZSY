import numpy as np
from constraints import calculate_insertion_cost

def construct_initial_solution(instance, E, C):
    electric_routes, remaining_customers = insertion_heuristic_electric(instance, E, C)
    conventional_routes = insertion_heuristic_conventional(instance, remaining_customers)
    initial_solution = {
        'electric': electric_routes,
        'conventional': conventional_routes,
        'remaining_customers': remaining_customers,
        'removed_customers': [],
        'instance': instance
    }
    return initial_solution


def insertion_heuristic_electric(instance, E, C):
    electric_routes = [[] for _ in E]
    remaining_customers = list(range(instance.n_customers))

    for i in range(len(electric_routes)):
        route = electric_routes[i]
        while remaining_customers:
            best_cost = float('inf')
            best_customer = None
            best_position = None

            for customer in remaining_customers:
                for position in range(len(route) + 1):
                    prev_customer = route[position - 1] if position > 0 else 'D0'
                    next_customer = route[position] if position < len(route) else 'D0'
                    cost = calculate_insertion_cost(instance, route, prev_customer, customer, next_customer)

                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_position = position

            if best_customer is not None:
                route.insert(best_position, best_customer)
                remaining_customers.remove(best_customer)

        electric_routes[i] = route

    return electric_routes, remaining_customers


def insertion_heuristic_conventional(instance, remaining_customers):
    routes = []
    while remaining_customers:
        route = [instance.O['id']]
        while remaining_customers:
            min_cost = float('inf')
            best_customer = None
            best_position = None
            for customer in remaining_customers:
                for i in range(1, len(route) + 1):
                    cost = calculate_insertion_cost(instance, route, i, customer)
                    if cost < min_cost:
                        min_cost = cost
                        best_customer = customer
                        best_position = i
            if best_customer is not None:
                route.insert(best_position, best_customer)
                remaining_customers.remove(best_customer)
            else:
                break
        route.append(instance.O_prime)
        routes.append(route)
    return routes


def insert_into_conventional(instance, routes, customer):
    best_route = None
    best_insertion_cost = float('inf')
    best_position = None

    for route in routes:
        for p in range(1, len(route)):
            insertion_cost = calculate_insertion_cost(instance, route, p, customer)
            if insertion_cost < best_insertion_cost:
                best_route = route
                best_insertion_cost = insertion_cost
                best_position = p

    if best_route is not None:
        best_route.insert(best_position, customer)
