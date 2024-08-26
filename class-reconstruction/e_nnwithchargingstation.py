from typing import List
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, CustomerLocation, ChargingStationLocation, DepotLocation
from c_constraints import Constraints
from a_read_instance import read_solomon_instance
import numpy as np
import matplotlib.pyplot as plt
import itertools


def calculate_scores(instance, E, C, lambda_param=0.5):
    # 计算电动车和燃油车的评分
    E_barycenter = np.mean(instance.customer_positions[E], axis=0) if E else instance.depot_position
    C_barycenter = np.mean(instance.customer_positions[C], axis=0) if C else instance.depot_position

    d_E = np.linalg.norm(instance.customer_positions - E_barycenter, axis=1)
    d_C = np.linalg.norm(instance.customer_positions - C_barycenter, axis=1)

    p_E = 11 - (1 + (d_E - np.min(d_E)) / (np.max(d_E) - np.min(d_E)) * 9) if np.max(d_E) != np.min(d_E) else np.full(d_E.shape, 10)
    pDist_C = 11 - (1 + (d_C - np.min(d_C)) / (np.max(d_C) - np.min(d_C)) * 9) if np.max(d_C) != np.min(d_C) else np.full(d_C.shape, 10)
    pQ = 11 - (1 + (instance.q_i[:instance.n] - np.min(instance.q_i[:instance.n])) / (np.max(instance.q_i[:instance.n]) - np.min(instance.q_i[:instance.n])) * 9) if np.max(instance.q_i[:instance.n]) != np.min(instance.q_i[:instance.n]) else np.full(instance.q_i[:instance.n].shape, 10)

    p_C = lambda_param * pDist_C[:instance.n] + (1 - lambda_param) * pQ

    return p_E, p_C


def clustering(instance, lambda_param=0.5):
    # 聚类算法，将客户分配到电动车和燃油车的服务区域
    E, C = [], []
    unassigned_customers = set(range(instance.n))

    while unassigned_customers:
        p_E, p_C = calculate_scores(instance, E, C, lambda_param)
        i_E = max(unassigned_customers, key=lambda i: p_E[i])
        i_C = max(unassigned_customers, key=lambda i: p_C[i])

        if i_E == i_C:
            E.append(i_E)
            unassigned_customers.remove(i_E)
        else:
            if p_E[i_E] > p_C[i_C]:
                E.append(i_E)
                unassigned_customers.remove(i_E)
            else:
                C.append(i_C)
                unassigned_customers.remove(i_C)

    return E, C


def nearest_neighbor_solution(instance, vehicle_type, customers, constraints):
    routes = []
    depot_index = instance.depot_index
    charging_stations = set(instance.charging_station_indices)  # 确保充电站单独存储
    max_attempts = 5  # 每个客户最多尝试5次

    while customers:
        current_route = [depot_index]
        remaining_capacity = instance.Q_e if vehicle_type == 'electric' else instance.Q_f
        remaining_battery = instance.B_star if vehicle_type == 'electric' else None
        current_location = depot_index
        charged_once = False
        total_energy_consumed = 0
        current_time = 0
        attempts = {}  # 记录每个客户的尝试次数

        while customers:
            nearest_customer, nearest_distance = None, float('inf')

            for customer in customers:
                if customer in attempts and attempts[customer] >= max_attempts:
                    continue  # 如果尝试次数达到最大值，跳过该客户

                # 计算当前车辆到该客户的距离
                distance = instance.d_ij[current_location, customer]
                travel_time = instance.t_ijk_e[current_location, customer] if vehicle_type == 'electric' else \
                instance.t_ijk_f[current_location, customer]
                arrival_time = current_time + travel_time

                # 确保只从客户列表中选择下一个访问的客户，而不是充电站
                if distance < nearest_distance and instance.q_i[customer] <= remaining_capacity and arrival_time <= \
                        instance.L_i[customer]:
                    nearest_customer = customer
                    nearest_distance = distance

            if nearest_customer is None:
                print("No valid customer found, ending this route.")
                break

            # 更新能量使用和时间
            if vehicle_type == 'electric':
                battery_usage = instance.L_ijk_e[current_location, nearest_customer]
                battery_usage_back_to_depot = instance.L_ijk_e[nearest_customer, depot_index]
                remaining_battery -= battery_usage
                total_energy_consumed += battery_usage
                travel_time = instance.t_ijk_e[current_location, nearest_customer]

                # 检查是否需要充电
                if remaining_battery < battery_usage_back_to_depot and not charged_once:
                    if charging_stations:
                        nearest_station = min(charging_stations, key=lambda s: instance.d_ij[current_location, s])
                        current_route.append(nearest_station)  # 使用 nearest_station 代替直接的 ID 来添加索引
                        print(f"Updated route after adding charging station: {current_route}")  # 调试打印
                        charging_stations.remove(nearest_station)
                        charge_time = (instance.B_star - remaining_battery) / instance.locations[
                            nearest_station].charging_rate
                        remaining_battery = instance.B_star
                        charged_once = True
                        current_time += charge_time
                        current_location = nearest_station

                        # 添加打印语句显示访问的充电站索引
                        print(f"Visited charging station: {nearest_station} (ID: {instance.locations[nearest_station].id})")
                        continue
                    else:
                        break
            else:
                travel_time = instance.t_ijk_f[current_location, nearest_customer]

            # 更新路径、容量和当前位置
            current_route.append(nearest_customer)
            customers.remove(nearest_customer)
            remaining_capacity -= instance.q_i[nearest_customer]
            current_location = nearest_customer

            # 打印当前客户的剩余载重
            print(f"Visiting customer {nearest_customer}: Remaining capacity {remaining_capacity}, Travel time {travel_time}")

            current_time += travel_time
            if not constraints.check_time_window(current_route):
                attempts[nearest_customer] = attempts.get(nearest_customer, 0) + 1
                if attempts[nearest_customer] < max_attempts:
                    print(f"Time window constraint violated at customer {nearest_customer}, trying to adjust route.")
                    customers.add(nearest_customer)
                    current_route.pop()
                    continue
                else:
                    print(f"Maximum attempts reached for customer {nearest_customer}, skipping this customer.")
                    break

        current_route.append(depot_index)
        print(f"Final route to be saved: {current_route}")  # 打印最终准备保存的路径
        routes.append((current_route[:], total_energy_consumed))  # 确保路径被正确保存

        print(f"Completed route: {current_route}, Energy consumed: {total_energy_consumed}")

    return routes


def construct_initial_solution(instance: CustomEVRPInstance):
    """
    生成电动车和燃油车的初始解。
    """
    constraints = Constraints(instance)
    electric_customers, fuel_customers = clustering(instance)
    electric_routes = nearest_neighbor_solution(instance, 'electric', set(electric_customers), constraints)
    fuel_routes = nearest_neighbor_solution(instance, 'fuel', set(fuel_customers), constraints)

    return electric_routes, fuel_routes


def plot_routes_with_stations(instance, electric_routes, fuel_routes, title):
    """
    绘制给定路径的路线图，包括充电桩位置，并区分电动车和燃油车的客户。
    """
    plt.figure(figsize=(10, 8))

    # 绘制配送中心的位置
    depot_position = np.array([instance.O.x, instance.O.y])
    plt.scatter(depot_position[0], depot_position[1], c='red', label='Depot', marker='s', zorder=3)

    # 绘制电动车客户
    electric_customers_positions = np.array([(instance.locations[loc_id].x, instance.locations[loc_id].y)
                                             for route, _ in electric_routes
                                             for loc_id in route if loc_id in instance.customer_indices])
    if electric_customers_positions.size > 0:
        plt.scatter(electric_customers_positions[:, 0], electric_customers_positions[:, 1],
                    c='blue', label='Electric Vehicle Customers', marker='o', zorder=2)

    # 绘制燃油车客户
    fuel_customers_positions = np.array([(instance.locations[loc_id].x, instance.locations[loc_id].y)
                                         for route, _ in fuel_routes
                                         for loc_id in route if loc_id in instance.customer_indices])
    if fuel_customers_positions.size > 0:
        plt.scatter(fuel_customers_positions[:, 0], fuel_customers_positions[:, 1],
                    c='orange', label='Fuel Vehicle Customers', marker='x', zorder=2)

    # 绘制充电桩
    charging_stations_positions = np.array(
        [(loc.x, loc.y) for loc in instance.locations if isinstance(loc, ChargingStationLocation)])
    plt.scatter(charging_stations_positions[:, 0], charging_stations_positions[:, 1], c='green',
                label='Charging Stations', marker='^', zorder=4)

    # 使用不同颜色绘制路径
    colors = itertools.cycle(plt.colormaps['tab20'].colors)

    # 绘制电动车路线
    for route, energy_consumed in electric_routes:
        route_positions = [instance.locations[loc_id].x for loc_id in route if loc_id in instance.customer_indices + [instance.depot_index]], \
            [instance.locations[loc_id].y for loc_id in route if loc_id in instance.customer_indices + [instance.depot_index]]
        color = next(colors)
        plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='--', color=color,
                 label=f'Electric Route: Energy Consumed {energy_consumed:.2f}', zorder=1)

    # 绘制燃油车路线
    for route, _ in fuel_routes:
        route_positions = [instance.locations[loc_id].x for loc_id in route if loc_id in instance.customer_indices + [instance.depot_index]], \
            [instance.locations[loc_id].y for loc_id in route if loc_id in instance.customer_indices + [instance.depot_index]]
        color = next(colors)
        plt.plot(route_positions[0], route_positions[1], marker='x', linestyle='-', color=color,
                 label='Fuel Vehicle Routes', zorder=1)

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.show()


def main():
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [CustomerLocation(**loc) if loc['type'] == 'c'
                 else ChargingStationLocation(**loc) if loc['type'] == 'f'
                 else DepotLocation(id=loc['id'], x=loc['x'], y=loc['y']) for loc in locations_data]

    instance = CustomEVRPInstance(locations, vehicles_data)
    electric_routes, fuel_routes = construct_initial_solution(instance)

    for i, (route, energy_consumed) in enumerate(electric_routes):
        print(f"Electric vehicle route {i + 1}: {route}, Energy consumed: {energy_consumed:.2f}")
    for i, (route, _) in enumerate(fuel_routes):
        print(f"Fuel vehicle route {i + 1}: {route}")

    plot_routes_with_stations(instance, electric_routes, fuel_routes, "Electric and Fuel Vehicle Routes with Charging Stations")


if __name__ == "__main__":
    main()
