from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance
from c_constraints import Constraints
import numpy as np

def calculate_scores(instance):
    # 计算电动车和燃油车的初始距离
    d_E = np.linalg.norm(instance.customer_positions - instance.depot_position, axis=1)
    d_C = np.linalg.norm(instance.customer_positions - instance.depot_position, axis=1)

    # 初始化变量
    d_E_min = np.min(d_E)
    d_E_max = np.max(d_E)
    d_C_min = np.min(d_C)
    d_C_max = np.max(d_C)
    q_min = np.min(instance.q_i)
    q_max = np.max(instance.q_i)

    # 设置lambda_param
    lambda_param = 0.5  # 可调参数

    # 计算电动车的评分
    if d_E_max != d_E_min:
        p_E = 11 - 1 + (d_E - d_E_min) / (d_E_max - d_E_min) * 9
    else:
        p_E = np.full(d_E.shape, 11 - 1)

    # 计算燃油车的评分
    if d_C_max != d_C_min:
        pDist_C = 11 - 1 + (d_C - d_C_min) / (d_C_max - d_C_min) * 9
    else:
        pDist_C = np.full(d_C.shape, 11 - 1)

    if q_max != q_min:
        pQ = 11 - 1 + (instance.q_i - q_min) / (q_max - q_min) * 9
    else:
        pQ = np.full(instance.q_i.shape, 11 - 1)

    p_C = lambda_param * pDist_C + (1 - lambda_param) * pQ

    return p_E, p_C

def clustering(instance):
    """
    将客户分配到电动车和燃油车的服务区域。
    """
    p_E, p_C = calculate_scores(instance)
    E, C = [], []
    unassigned_customers = set(range(1, instance.n + 1))

    while unassigned_customers:
        i_E = max(unassigned_customers, key=lambda i: p_E[i - 1])
        i_C = max(unassigned_customers, key=lambda i: p_C[i - 1])

        if i_E == i_C:
            E.append(i_E)
            unassigned_customers.remove(i_E)
        else:
            if p_E[i_E - 1] > p_C[i_C - 1]:
                E.append(i_E)
                unassigned_customers.remove(i_E)
            else:
                C.append(i_C)
                unassigned_customers.remove(i_C)

        remaining_customers = list(unassigned_customers)
        if remaining_customers:
            # 将索引从 1 基转换为 0 基
            remaining_customers_0_based = [i - 1 for i in remaining_customers]
            E_positions = instance.customer_positions[[i - 1 for i in E]]
            C_positions = instance.customer_positions[[i - 1 for i in C]]

            E_barycenter = np.mean(E_positions, axis=0) if E_positions.size else instance.depot_position
            C_barycenter = np.mean(C_positions, axis=0) if C_positions.size else instance.depot_position

            d_E = np.linalg.norm(instance.customer_positions[remaining_customers_0_based] - E_barycenter, axis=1)
            d_C = np.linalg.norm(instance.customer_positions[remaining_customers_0_based] - C_barycenter, axis=1)

            d_E_min = np.min(d_E)
            d_E_max = np.max(d_E)
            d_C_min = np.min(d_C)
            d_C_max = np.max(d_C)

            # 初始化pDist_C和pQ变量
            pDist_C = np.full(p_C.shape, 11 - 1)
            pQ = np.full(p_C.shape, 11 - 1)

            if d_E_max != d_E_min:
                p_E[remaining_customers_0_based] = 11 - 1 + (d_E - d_E_min) / (d_E_max - d_E_min) * 9
            else:
                p_E[remaining_customers_0_based] = 11 - 1

            if d_C_max != d_C_min:
                pDist_C[remaining_customers_0_based] = 11 - 1 + (d_C - d_C_min) / (d_C_max - d_C_min) * 9
            else:
                pDist_C[remaining_customers_0_based] = 11 - 1

            q_min = np.min(instance.q_i)
            q_max = np.max(instance.q_i)

            if q_max != q_min:
                pQ[remaining_customers_0_based] = 11 - 1 + (instance.q_i[remaining_customers_0_based] - q_min) / (q_max - q_min) * 9
            else:
                pQ[remaining_customers_0_based] = 11 - 1

            lambda_param = 0.5  # 可调参数
            p_C[remaining_customers_0_based] = lambda_param * pDist_C[remaining_customers_0_based] + (1 - lambda_param) * pQ[remaining_customers_0_based]

    return E, C

def calculate_total_cost(instance, current_route, next_customer, vehicle_type):
    """
    计算将下一个客户加入当前路径后的总成本。
    """
    new_route = current_route + [next_customer]

    fixed_cost = instance.calculate_fixed_cost(vehicle_type)
    transport_cost = instance.calculate_transport_cost(new_route, vehicle_type)
    loss_cost = instance.calculate_loss_cost(new_route, vehicle_type)
    charging_cost = instance.calculate_charging_cost(new_route, vehicle_type)
    time_window_penalty = instance.calculate_time_window_penalty(new_route, vehicle_type)
    carbon_cost = instance.calculate_carbon_cost(new_route, vehicle_type)

    total_cost = (
        fixed_cost +
        transport_cost +
        loss_cost +
        charging_cost +
        time_window_penalty +
        carbon_cost
    )

    return total_cost

def insert_charging_station(instance, current_route, remaining_battery, vehicle_type):
    """
    尝试在当前路径中插入一个充电桩，并返回插入后的路径和总成本。
    """
    best_cost = float('inf')
    best_route = None
    best_station = None

    for i, customer in enumerate(current_route):
        for charging_station in instance.charging_stations:
            new_route = current_route[:i + 1] + [charging_station] + current_route[i + 1:]

            required_battery = instance.calculate_battery_usage(new_route[i + 1:])
            charging_needed = max(0, required_battery - remaining_battery)

            total_cost = calculate_total_cost(instance, current_route, charging_station, vehicle_type) + \
                         instance.calculate_transport_cost(new_route, vehicle_type) + \
                         instance.calculate_charging_cost(charging_needed, vehicle_type)

            if total_cost < best_cost:
                best_cost = total_cost
                best_route = new_route
                best_station = charging_station

    new_remaining_battery = remaining_battery + charging_needed

    return best_route, best_cost, new_remaining_battery

def is_feasible(instance, constraints_checker, current_route, next_customer, vehicle_type):
    """
    检查将 next_customer 添加到 current_route 后，路径是否仍然满足约束条件。
    """
    new_route = current_route + [next_customer]

    if not constraints_checker.check_node_visit(new_route, vehicle_type):
        return False

    if not constraints_checker.check_load_balance(new_route, vehicle_type):
        return False

    if not constraints_checker.check_time_window(new_route, vehicle_type):
        return False

    if vehicle_type == 'electric' and not constraints_checker.check_battery(new_route):
        return False

    return True

def greedy_initial_solution_with_partial_charging(instance, vehicle_type, customers):
    """
    使用贪婪算法为电动车生成初始解，并考虑部分充电策略。
    """
    routes = []
    unassigned_customers = set(customers)
    constraints_checker = Constraints(instance)
    remaining_battery = instance.B_star  # 初始电量为满电

    while unassigned_customers:
        current_route = []
        remaining_capacity = instance.Q_e

        while unassigned_customers:
            best_customer = None
            best_cost = float('inf')

            for customer in unassigned_customers:
                if instance.q_i[customer - 1] <= remaining_capacity:
                    total_cost = calculate_total_cost(instance, current_route, customer, vehicle_type)

                    if total_cost < best_cost and is_feasible(instance, constraints_checker, current_route, customer, vehicle_type):
                        best_cost = total_cost
                        best_customer = customer

            if best_customer is None:
                break

            current_route.append(best_customer)
            unassigned_customers.remove(best_customer)
            remaining_capacity -= instance.q_i[best_customer - 1]
            remaining_battery -= instance.calculate_battery_usage(best_customer)

            if remaining_battery < instance.minimum_battery_threshold:
                current_route, best_cost, remaining_battery = insert_charging_station(instance, current_route,
                                                                                      remaining_battery, vehicle_type)

        if len(current_route) > 1:
            routes.append(current_route)

    return routes

def greedy_initial_solution_for_fuel(instance, vehicle_type, customers):
    """
    使用贪婪算法为燃油车生成初始解。
    """
    routes = []
    unassigned_customers = set(customers)
    constraints_checker = Constraints(instance)

    while unassigned_customers:
        current_route = []
        remaining_capacity = instance.Q_f

        while unassigned_customers:
            best_customer = None
            best_cost = float('inf')

            for customer in unassigned_customers:
                if instance.q_i[customer - 1] <= remaining_capacity:
                    total_cost = calculate_total_cost(instance, current_route, customer, vehicle_type)

                    if total_cost < best_cost and is_feasible(instance, constraints_checker, current_route, customer, vehicle_type):
                        best_cost = total_cost
                        best_customer = customer

            if best_customer is None:
                break

            current_route.append(best_customer)
            unassigned_customers.remove(best_customer)
            remaining_capacity -= instance.q_i[best_customer - 1]

        if current_route:
            routes.append(current_route)

    return routes

def construct_initial_solution_with_partial_charging(instance: CustomEVRPInstance):
    """
    聚类后生成电动车和燃油车的初始解。
    """
    electric_customers, fuel_customers = clustering(instance)

    electric_routes = greedy_initial_solution_with_partial_charging(instance, 'electric', electric_customers)
    fuel_routes = greedy_initial_solution_for_fuel(instance, 'fuel', fuel_customers)

    return electric_routes, fuel_routes
