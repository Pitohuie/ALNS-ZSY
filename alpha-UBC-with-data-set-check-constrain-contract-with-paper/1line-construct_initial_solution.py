import os
import numpy as np
import matplotlib.pyplot as plt
from read_instance import read_solomon_instance
from evrp_instance import CustomEVRPInstance
from evrp_state import EVRPState


def clustering(instance, lambda_=0.5):
    N = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    R.extend([instance.O, instance.O_prime])  # 将起点和终点也作为充电桩的一部分
    V = N + R
    E, C = set(), set()
    s = instance.O

    # Initial barycentre
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
        p_C = {i: calculate_score_C(d_C[i], d_C_min, d_C_max, q.get(i, 0), q_min, q_max, lambda_) for i in N if
               i not in E and i not in C}

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

        # Recalculate barycentre
        if E:
            b_e = np.mean([instance.locations[i]['x'] for i in E]), np.mean([instance.locations[i]['y'] for i in E])
        if C:
            b_c = np.mean([instance.locations[i]['x'] for i in C]), np.mean([instance.locations[i]['y'] for i in C])

    # Ensure s is in the sets then remove
    if s in E:
        E.remove(s)
    if s in C:
        C.remove(s)

    # Ensure indices are in range
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
                    if vehicle_type == 'conventional' and can_insert_conventional(instance, route, i, node,
                                                                                  current_load):
                        try:
                            cost = calculate_insertion_cost(instance, route, i, node)
                            if cost < best_cost:
                                best_cost = cost
                                best_node = node
                                best_pos = i
                        except IndexError as e:
                            print(f"Index error when inserting customer {node}: {e}")
                    elif vehicle_type == 'electric' and can_insert_electric(instance, route, i, node, current_load,
                                                                            current_battery):
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

        print(f"Route completed: {route}")  # Debug: Print each completed route_in
        routes.append(route)

    print(f"Total routes generated: {len(routes)}")  # Debug: Print the total number of routes generated
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

        if instance.locations[customer]['type'] == 'c' and (current_time > instance.time_window_end[
            customer - 1] or current_battery < instance.B_star * instance.soc_min):
            return False

    return True


def sequential_insertion_heuristic(instance):
    # Step 1: 聚类
    C, E = clustering(instance)

    print(f"Cluster C (conventional vehicles): {C}")
    print(f"Cluster E (electric vehicles): {E}")

    # Step 2: 插入启发式 for conventional vehicles
    customer_routes = insertion_heuristic(instance, C, vehicle_type='conventional')

    # Step 3: 更新E if there are unserved customers
    unserved_customers = set(C) - set([node for route in customer_routes for node in route])
    print("Unserved customers:", unserved_customers)
    if unserved_customers:
        E.update(unserved_customers)

    # Step 4: 尝试将未服务的客户插入到电动车路径中
    if unserved_customers:
        electric_vehicle_insertion_failed = False
        try:
            charging_routes = insertion_heuristic(instance, E, vehicle_type='electric')
        except IndexError:
            electric_vehicle_insertion_failed = True
            print("Error inserting into electric vehicle routes, retrying with conventional vehicles")

        if electric_vehicle_insertion_failed:
            # 如果电动车插入失败，将其直接放入常规车辆路径中
            for customer in unserved_customers:
                customer_routes = insert_into_existing_route(instance, customer_routes, customer)
    else:
        charging_routes = insertion_heuristic(instance, E, vehicle_type='electric')

    # Step 5: 合并结果并验证
    final_routes = customer_routes + charging_routes
    initial_state = EVRPState(final_routes, instance)
    final_state = verify_and_repair(instance, initial_state)

    return final_state


def insert_into_existing_route(instance, routes, customer):
    """
    尝试将未服务的客户插入到现有的常规车辆路径中。
    """
    best_cost = float('inf')
    best_route_idx = -1
    best_pos = -1

    for route_idx, route in enumerate(routes):
        for i in range(len(route) + 1):
            if can_insert_conventional(instance, route, i, customer, instance.customer_demand[customer - 1]):
                try:
                    cost = calculate_insertion_cost(instance, route, i, customer)
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = route_idx
                        best_pos = i
                except IndexError as e:
                    print(f"Index error when inserting customer {customer}: {e}")

    if best_route_idx == -1 or best_pos == -1:
        raise RuntimeError(f"Failed to insert unserved customer {customer} into any existing route.")

    routes[best_route_idx].insert(best_pos, customer)
    return routes


def plot_solution(state, instance):
    fig, ax = plt.subplots()

    for i, loc in enumerate(instance.locations):
        if loc['type'] == 'c':
            ax.plot(loc['x'], loc['y'], 'bo')  # 客户点
        elif loc['type'] == 'f':
            ax.plot(loc['x'], loc['y'], 'ro')  # 充电站点

    depot = instance.locations[instance.O]
    ax.plot(depot['x'], depot['y'], 'gs', markersize=10)  # 配送中心

    # 绘制路径
    for route in state.routes:
        print(f"Route: {route}")  # Debug: Print each route_in
        x_coords = [instance.locations[node]['x'] for node in route]
        y_coords = [instance.locations[node]['y'] for node in route]
        print(f"x_coords: {x_coords}")  # Debug: Print x coordinates of the route_in
        print(f"y_coords: {y_coords}")  # Debug: Print y coordinates of the route_in
        ax.plot(x_coords, y_coords, 'k-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('EVRP Solution')
    plt.show()


# 设定为当前工作目录
os.chdir('D:\\2024\\ZSY-ALNS\\pythonProject1\\alpha-UBC-with-data-set-check-constrain')  # 请根据实际路径进行调整

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
