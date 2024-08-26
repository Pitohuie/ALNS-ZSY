# greedy_algorithm.py

from constraints import Constraints


def calculate_total_cost(instance, current_route, next_customer, vehicle_type):
    """
    计算将下一个客户加入当前路径后的总成本。

    :param instance: CustomEVRPInstance 实例
    :param current_route: 当前路径
    :param next_customer: 待加入的下一个客户
    :param vehicle_type: 车辆类型 ('electric' 或 'fuel')
    :return: 总成本
    """
    # 创建一个新的路径副本，模拟将下一个客户加入路径后的情况
    new_route = current_route + [next_customer]

    # 计算各种成本
    fixed_cost = instance.calculate_fixed_cost(vehicle_type)
    transport_cost = instance.calculate_transport_cost(new_route, vehicle_type)
    loss_cost = instance.calculate_loss_cost(new_route, vehicle_type)
    charging_cost = instance.calculate_charging_cost(new_route, vehicle_type)
    time_window_penalty = instance.calculate_time_window_penalty(new_route, vehicle_type)
    carbon_cost = instance.calculate_carbon_cost(new_route, vehicle_type)

    # 计算总成本
    total_cost = (
            fixed_cost +
            transport_cost +
            loss_cost +
            charging_cost +
            time_window_penalty +
            carbon_cost
    )

    return total_cost


def is_feasible(instance, constraints_checker, current_route, next_customer, vehicle_type):
    """
    检查将 next_customer 添加到 current_route 后，路径是否仍然满足约束条件。
    """
    # 创建一个新的路线副本
    new_route = current_route + [next_customer]

    # 检查各项约束
    if not constraints_checker.check_node_visit(new_route, vehicle_type):
        return False

    if not constraints_checker.check_load_balance(new_route, vehicle_type):
        return False

    if not constraints_checker.check_time_window(new_route, vehicle_type):
        return False

    if vehicle_type == 'electric' and not constraints_checker.check_battery(new_route):
        return False

    return True


def greedy_initial_solution_with_costs(instance, vehicle_type, customers):
    """
    使用基于多个成本的贪婪算法生成初始解。

    :param instance: CustomEVRPInstance 实例
    :param vehicle_type: 车辆类型 ('electric' 或 'fuel')
    :param customers: 当前车辆类型对应的客户集合
    :return: 生成的初始解，格式为 routes 列表，每个子列表是一条路线
    """
    routes = []
    unassigned_customers = set(customers)
    constraints_checker = Constraints(instance)  # 初始化约束检查器

    while unassigned_customers:
        current_route = []
        remaining_capacity = instance.Q_e if vehicle_type == 'electric' else instance.Q_f

        while unassigned_customers:
            best_customer = None
            best_cost = float('inf')

            for customer in unassigned_customers:
                if instance.q_i[customer - 1] <= remaining_capacity:
                    total_cost = calculate_total_cost(instance, current_route, customer, vehicle_type)

                    # 检查是否满足约束，并且成本最小
                    if total_cost < best_cost and is_feasible(instance, constraints_checker, current_route, customer,
                                                              vehicle_type):
                        best_cost = total_cost
                        best_customer = customer

            if best_customer is None:
                break  # 没有合适的客户可以添加

            current_route.append(best_customer)
            unassigned_customers.remove(best_customer)
            remaining_capacity -= instance.q_i[best_customer - 1]

        if current_route:
            routes.append(current_route)

    return routes
