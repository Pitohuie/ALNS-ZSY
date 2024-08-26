import os
import numpy as np
import matplotlib.pyplot as plt
from read_instance import read_solomon_instance
from evrp_instance import CustomEVRPInstance
from evrp_state import EVRPState

def clustering(instance, lambda_=0.5):
    N = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    R.extend([instance.O, instance.O_prime])
    V = N + R
    E, C = set(), set()
    s = instance.O

    b_e = np.mean([instance.locations[i]['x'] for i in N]), np.mean([instance.locations[i]['y'] for i in N])
    b_c = b_e

    while len(E) + len(C) < len(N):
        d_E = {i: np.linalg.norm([instance.locations[i]['x'] - b_e[0], instance.locations[i]['y'] - b_e[1]]) for i in N}
        d_C = {i: np.linalg.norm([instance.locations[i]['x'] - b_c[0], instance.locations[i]['y'] - b_c[1]]) for i in N}

        d_E_min, d_E_max = min(d_E.values()), max(d_E.values())
        d_C_min, d_C_max = min(d_C.values()), max(d_C.values())

        q = {i: instance.customer_demand[i - 1] for i in N if i - 1 < len(instance.customer_demand)}
        q_min, q_max = min(q.values()), max(q.values())

        p_E = {i: calculate_score_E(d_E[i], d_E_min, d_E_max) for i in N if i not in E and i not in C}
        p_C = {i: calculate_score_C(d_C[i], d_C_min, d_C_max, q.get(i, 0), q_min, q_max, lambda_) for i in N if i not in E and i not in C}

        i_E = max(p_E, key=p_E.get)
        i_C = max(p_C, key=p_C.get)

        if i_E == i_C:
            E.add(i_E)
            C.add(i_C)
        else:
            if p_E[i_E] > p_C[i_C]:
                E.add(i_E)
            else:
                C.add(i_C)

        if E:
            b_e = np.mean([instance.locations[i]['x'] for i in E]), np.mean([instance.locations[i]['y'] for i in E])
        if C:
            b_c = np.mean([instance.locations[i]['x'] for i in C]), np.mean([instance.locations[i]['y'] for i in C])

    if s in E:
        E.remove(s)
    if s in C:
        C.remove(s)

    max_index = len(instance.time_window_start)
    E = {i for i in E if i < max_index}
    C = {i for i in C if i < max_index}

    return list(E), list(C)

def calculate_score_E(d_i, d_min, d_max):
    return 1 + (d_i - d_min) / (d_max - d_min) * 9

def calculate_score_C(d_i, d_min, d_max, q_i, q_min, q_max, lambda_):
    pDist_C_i = 1 + (d_i - d_min) / (d_max - d_min) * 9
    pQ_i = 1 + (q_i - q_min) / (q_max - q_min) * 9
    return lambda_ * pDist_C_i + (1 - lambda_) * pQ_i

def calculate_insertion_cost(instance, route, idx, customer):
    distances = instance.distance_matrix
    pred = instance.O if idx == 0 else route[idx - 1]
    succ = instance.O_prime if idx == len(route) else route[idx]

    if pred >= len(distances) or customer >= len(distances) or succ >= len(distances):
        raise IndexError(f"Index out of bounds: pred={pred}, customer={customer}, succ={succ}")

    return distances[pred][customer] + distances[customer][succ] - distances[pred][succ]

def apply_constraints(constraints):
    for constraint in constraints:
        if not constraint:
            return False
    return True

def node_visit_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    M = instance.M

    max_index = min(len(instance.x), len(instance.locations))

    constraints.append(
        sum(instance.x[0, j, k] for j in C + R if j < max_index for k in K if k < max_index) ==
        sum(instance.x[j, instance.O_prime, k] for j in C + R if j < max_index for k in K if k < max_index)
    )

    for i in C:
        if i < max_index:
            constraints.append(
                sum(instance.x[i, j, k] for j in N if j != i and j < max_index for k in K if k < max_index) == 1
            )

    for j in C:
        if j < max_index:
            for k in K:
                if k < max_index:
                    constraints.append(
                        sum(instance.x[i, j, k] for i in N if i != j and i < max_index) ==
                        sum(instance.x[j, i, k] for i in N if i != j and i < max_index)
                    )

    for k in K_e:
        if k < max_index:
            constraints.append(
                sum(instance.x[i, j, k] for i in N if i < max_index for j in R if j < max_index) <= instance.sigma
            )

    return constraints

def load_balance_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    E = [(i, j) for i in N for j in N if i != j]
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    K_f = list(range(instance.k_e, instance.k_e + instance.k_f))
    M = instance.M
    Q_e = instance.Q_e
    Q_f = instance.Q_f
    q = instance.customer_demand

    max_index = min(len(instance.u), len(instance.locations))

    for (i, j) in E:
        if i < max_index and j < max_index:
            for k in K_e:
                if k < max_index:
                    constraints.append(instance.u[i, j, k] <= Q_e)
            for k in K_f:
                if k < max_index:
                    constraints.append(instance.u[i, j, k] <= Q_f)

    for j in C:
        if j < max_index and j < len(q):
            for k in K:
                if k < max_index:
                    constraints.append(
                        sum(instance.u[i, j, k] for i in N if i != j and i < max_index) -
                        sum(instance.u[j, i, k] for i in N if i != j and i < max_index) +
                        M * (1 - sum(instance.x[i, j, k] for i in N if i != j and i < max_index)) >= q[j]
                    )

    for k in K:
        if k < max_index:
            constraints.append(
                sum(instance.u[i, instance.O_prime, k] for i in C + R if i < max_index) == 0
            )

    return constraints

def time_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    M = instance.M
    t = instance.travel_time_matrix
    s = instance.service_time
    w = np.zeros((instance.N, len(K)))  # 确保w是二维数组
    E = [loc['ready_time'] for loc in instance.locations]
    L = [loc['due_date'] for loc in instance.locations]

    max_index = min(len(instance.x), len(instance.locations), len(t))

    for k in K:
        if k < len(instance.b):  # 添加边界检查
            constraints.append(instance.b[0, k] >= E[0])

    for j in C:
        if j < max_index:
            for k in K:
                if k < len(instance.b) and j < len(instance.a):  # 添加边界检查
                    constraints.append(
                        instance.b[0, k] + t[0, j] * instance.x[0, j, k] - M * (1 - instance.x[0, j, k]) <= instance.a[j, k]
                    )

    for i in C:
        if i < max_index:
            for j in N:
                if j != i and j < max_index:
                    for k in K:
                        if k < len(instance.a) and i < len(instance.a) and j < len(instance.a):  # 添加边界检查
                            constraints.append(
                                instance.a[i, k] + t[i, j] * instance.x[i, j, k] + s[i] + w[i, k] - M * (1 - instance.x[i, j, k]) <= instance.a[j, k]
                            )

    for i in N:
        if i < max_index:
            for k in K:
                if k < len(instance.b):  # 添加边界检查
                    constraints.append(
                        instance.b[i, k] + t[i, instance.O_prime] * instance.x[i, instance.O_prime, k] - M * (1 - instance.x[i, instance.O_prime, k]) <= L[0]
                    )

    for i in C:
        if i < max_index:
            for k in K:
                if k < len(instance.a):  # 添加边界检查
                    constraints.append(
                        instance.b[i, k] == instance.a[i, k] + s[i] + w[i, k]
                    )

    for i in R:
        if i < max_index:
            for k in K_e:
                if k < len(instance.b):  # 添加边界检查
                    constraints.append(
                        instance.b[i, k] == instance.a[i, k] + M
                    )

    for i in C:
        if i < max_index:
            for k in K_e:
                if k < len(instance.a):  # 添加边界检查
                    constraints.append(
                        E[i] <= instance.a[i, k] <= L[i]
                    )

    return constraints

def battery_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K_e = list(range(instance.k_e))
    M = instance.M
    L_ijk = instance.L_ijk
    B_star = instance.B_star
    soc_min = instance.soc_min

    for i in N:
        for j in N:
            if j != i:
                for k in K_e:
                    constraints.append(
                        instance.B[i, k] - L_ijk[i, j] * instance.x[i, j, k] + B_star * (1 - instance.x[i, j, k]) >= instance.B[j, k]
                    )

    for k in K_e:
        constraints.append(
            instance.B[0, k] == B_star
        )

    for i in N:
        for k in K_e:
            constraints.append(
                instance.B[i, k] >= B_star * soc_min
            )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.B[r, k] == B_star
            )

    return constraints

def variable_constraints(instance):
    constraints = []

    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))

    for (i, j) in [(i, j) for i in N for j in N if i != j]:
        for k in K:
            constraints.append(instance.u[i, j, k] >= 0)
            if k in range(instance.k_e):
                constraints.append(instance.f[i, j, k] >= 0)
            constraints.append(instance.x[i, j, k] == 0 or instance.x[i, j, k] == 1)

    return constraints

def insertion_heuristic(instance, nodes, vehicle_type='conventional'):
    routes = []
    unvisited = set(nodes)
    visited_customers = set()

    while unvisited:
        if vehicle_type == 'conventional':
            route, current_load, current_time = initialize_route_conventional(instance, unvisited)
        else:
            route, current_load, current_time, current_battery = initialize_route_electric(instance, unvisited)

        while unvisited:
            best_cost = float('inf')
            best_node = None
            best_pos = None

            for node in unvisited:
                if node in visited_customers and instance.locations[node]['type'] == 'c':
                    continue

                for i in range(len(route) + 1):
                    if vehicle_type == 'conventional' and can_insert_conventional(instance, route, i, node, current_load):
                        try:
                            cost = calculate_insertion_cost(instance, route, i, node)
                            if cost < best_cost:
                                best_cost = cost
                                best_node = node
                                best_pos = i
                        except IndexError as e:
                            print(f"Index error when inserting customer {node}: {e}")
                    elif vehicle_type == 'electric' and can_insert_electric(instance, route, i, node, current_load, current_battery):
                        try:
                            cost = calculate_insertion_cost(instance, route, i, node)
                            if cost < best_cost:
                                best_cost = cost
                                best_node = node
                                best_pos = i
                        except IndexError as e:
                            print(f"Index error when inserting customer {node}: {e}")

            if best_node is None:
                break

            if best_pos > len(route):
                raise IndexError(f"Insertion position index out of range: {best_pos}, route length: {len(route)}")

            if best_node >= len(instance.locations):
                raise IndexError(f"Insertion customer index out of range: {best_node}")

            # 插入之前应用约束
            constraints = [
                node_visit_constraints(instance),
                load_balance_constraints(instance),
                time_constraints(instance),
                battery_constraints(instance),
                variable_constraints(instance)
            ]
            if apply_constraints(constraints):
                try:
                    route.insert(best_pos, best_node)
                    unvisited.remove(best_node)
                    if instance.locations[best_node]['type'] == 'c':
                        visited_customers.add(best_node)
                    current_load += instance.customer_demand[best_node - 1]
                    if best_pos > 0:
                        current_time += instance.travel_time_matrix[route[best_pos - 1]][best_node]
                        if vehicle_type == 'electric':
                            current_battery -= instance.L_ijk[route[best_pos - 1]][best_node]
                except IndexError:
                    print(f"Failed to insert customer {best_node} in electric vehicle route, switching to conventional")
                    if vehicle_type == 'electric':
                        unvisited.add(best_node)
                    else:
                        raise

        routes.append(route)

    return routes

def can_insert_conventional(instance, route, idx, customer, current_load):
    new_load = current_load + instance.customer_demand[customer - 1]
    return new_load <= instance.Q_f

def can_insert_electric(instance, route, idx, customer, current_load, current_battery):
    new_load = current_load + instance.customer_demand[customer - 1]
    if new_load > instance.Q_e:
        return False

    if idx > 0 and (route[idx - 1] >= len(instance.L_ijk) or customer >= len(instance.L_ijk)):
        print(f"Electric insertion error: idx={idx}, route[idx - 1]={route[idx - 1]}, customer={customer}")
        raise IndexError(f"Index out of bounds: route[idx - 1]={route[idx - 1]}, customer={customer}")

    new_battery = current_battery - instance.L_ijk[route[idx - 1]][customer] if idx > 0 else instance.B_star
    return new_battery >= instance.B_star * instance.soc_min

def initialize_route_conventional(instance, unvisited):
    valid_unvisited = {i for i in unvisited if 0 <= i < len(instance.time_window_start)}
    if not valid_unvisited:
        raise ValueError("No valid unvisited nodes")

    start_node = min(valid_unvisited, key=lambda i: instance.time_window_start[i])
    unvisited.remove(start_node)
    print(f"Initializing conventional route with start node {start_node}")
    return [instance.O, start_node, instance.O_prime], instance.customer_demand[start_node], 0

def initialize_route_electric(instance, unvisited):
    valid_unvisited = {i for i in unvisited if 0 <= i < len(instance.time_window_start)}
    if not valid_unvisited:
        raise ValueError("No valid unvisited nodes")

    start_node = min(valid_unvisited, key=lambda i: instance.time_window_start[i])
    unvisited.remove(start_node)
    print(f"Initializing electric route with start node {start_node}")
    return [instance.O, start_node, instance.O_prime], instance.customer_demand[start_node], 0, instance.B_star

def verify_and_repair(instance, state):
    for route in state.routes:
        current_time = 0
        for i in range(1, len(route)):
            customer = route[i]
            prev_customer = route[i - 1]
            current_time += instance.travel_time_matrix[prev_customer][customer]

            if instance.locations[customer]['type'] == 'c' and customer - 1 >= len(instance.time_window_end):
                raise IndexError(f"Customer index {customer - 1} is out of bounds for time window end array.")

            if instance.locations[customer]['type'] == 'c' and current_time > instance.time_window_end[customer - 1]:
                return repair_solution(instance, state)

    return state

def repair_solution(instance, state):
    for route in state.routes:
        while not verify_route(instance, route):
            route.pop()

    return state

def verify_route(instance, route):
    current_time = 0
    current_battery = instance.B_star

    for i in range(1, len(route)):
        customer = route[i]
        prev_customer = route[i - 1]
        current_time += instance.travel_time_matrix[prev_customer][customer]
        current_battery -= instance.L_ijk[prev_customer][customer]

        if instance.locations[customer]['type'] == 'c' and customer - 1 >= len(instance.time_window_end):
            raise IndexError(f"Customer index {customer - 1} is out of bounds for time window end array.")

        if instance.locations[customer]['type'] == 'c' and (current_time > instance.time_window_end[customer - 1] or current_battery < instance.B_star * instance.soc_min):
            return False

    return True

def sequential_insertion_heuristic(instance):
    C, E = clustering(instance)

    print(f"Cluster C (conventional vehicles): {C}")
    print(f"Cluster E (electric vehicles): {E}")

    customer_routes = insertion_heuristic(instance, C, vehicle_type='conventional')

    unserved_customers = set(C) - set([node for route in customer_routes for node in route])
    if unserved_customers:
        E.extend(unserved_customers)

    electric_vehicle_insertion_failed = False
    try:
        charging_routes = insertion_heuristic(instance, E, vehicle_type='electric')
    except IndexError:
        electric_vehicle_insertion_failed = True
        print("Error inserting into electric vehicle routes, retrying with conventional vehicles")

    if electric_vehicle_insertion_failed:
        charging_routes = insertion_heuristic(instance, E, vehicle_type='conventional')

    final_routes = customer_routes + charging_routes

    initial_state = EVRPState(final_routes, instance)

    final_state = verify_and_repair(instance, initial_state)

    return final_state

def plot_solution(state, instance):
    fig, ax = plt.subplots()

    for i, loc in enumerate(instance.locations):
        if loc['type'] == 'c':
            ax.plot(loc['x'], loc['y'], 'bo')
        elif loc['type'] == 'f':
            ax.plot(loc['x'], loc['y'], 'ro')

    depot = instance.locations[instance.O]
    ax.plot(depot['x'], depot['y'], 'gs', markersize=10)

    for route in state.routes:
        print(f"Route: {route}")
        x_coords = [instance.locations[node]['x'] for node in route]
        y_coords = [instance.locations[node]['y'] for node in route]
        print(f"x_coords: {x_coords}")
        print(f"y_coords: {y_coords}")
        ax.plot(x_coords, y_coords, 'k-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('EVRP Solution')
    plt.show()

os.chdir('D:\\2024\\ZSY-ALNS\\pythonProject1\\alpha-UBC-with-data-set-check-constrain')

print("Current working directory:", os.getcwd())

file_path = 'evrptw_instances/c101_21.txt'
locations, vehicles = read_solomon_instance(file_path)
instance = CustomEVRPInstance(locations, vehicles)
print("Locations:", locations)
print("Vehicles:", vehicles)

initial_state = sequential_insertion_heuristic(instance)

print("Initial Routes:")
for route in initial_state.routes:
    print(route)

plot_solution(initial_state, instance)
