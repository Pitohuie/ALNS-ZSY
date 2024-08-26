from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from a_read_instance import read_solomon_instance
import numpy as np
import matplotlib.pyplot as plt
import itertools


def calculate_scores(instance, E, C, lambda_param=0.5):
    """
    计算电动车和常规车辆的评分。
    """
    E_barycenter = np.mean(instance.customer_positions[[i - 1 for i in E]], axis=0) if E else instance.depot_position
    C_barycenter = np.mean(instance.customer_positions[[i - 1 for i in C]], axis=0) if C else instance.depot_position

    d_E = np.linalg.norm(instance.customer_positions - E_barycenter, axis=1)
    d_C = np.linalg.norm(instance.customer_positions - C_barycenter, axis=1)

    # 计算电动车评分 p_E
    d_E_min, d_E_max = np.min(d_E), np.max(d_E)
    if d_E_max != d_E_min:
        p_E = 11 - (1 + (d_E - d_E_min) / (d_E_max - d_E_min) * 9)
    else:
        p_E = np.full(d_E.shape, 11 - 1)

    # 计算燃油车评分 p_C
    d_C_min, d_C_max = np.min(d_C), np.max(d_C)
    if d_C_max != d_C_min:
        pDist_C = 11 - (1 + (d_C - d_C_min) / (d_C_max - d_C_min) * 9)
    else:
        pDist_C = np.full(d_C.shape, 11 - 1)

    q_min, q_max = np.min(instance.q_i[:instance.n]), np.max(instance.q_i[:instance.n])
    if q_max != q_min:
        pQ = 11 - (1 + (instance.q_i[:instance.n] - q_min) / (q_max - q_min) * 9)
    else:
        pQ = np.full(instance.q_i[:instance.n].shape, 11 - 1)

    p_C = lambda_param * pDist_C[:instance.n] + (1 - lambda_param) * pQ

    return p_E, p_C


def clustering(instance, lambda_param=0.5):
    """
    聚类算法，将客户分配到电动车和常规车辆的服务区域。
    """
    E, C = [], []
    unassigned_customers = set(range(1, instance.n + 1))

    while unassigned_customers:
        p_E, p_C = calculate_scores(instance, E, C, lambda_param)

        # 找到未分配客户中评分最高的客户
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

    return E, C


def nearest_neighbor_solution(instance, vehicle_type, customers):
    """
    使用最近邻算法生成车辆的初始解，并且管理电量，在电量不足时加入充电站。
    每条路径最多只允许充一次电，每个充电站只能在路径上使用一次。
    """
    global battery_usage, waiting_time, arrival_time
    routes = []
    battery_capacity = instance.B_star  # 电动车初始电量为满电
    instance.calculate_energy_consumption_factors()  # 计算能耗系数
    charging_stations = set(instance.type_to_indices['f'])  # 使用类型字典获取充电站的索引
    depot_index = instance.type_to_indices['d'][0]  # 获取配送中心索引
    customer_indices = instance.type_to_indices['c']  # 获取客户索引

    while customers:
        current_route = [depot_index]  # 路线从配送中心开始
        remaining_capacity = instance.Q_e if vehicle_type == 'electric' else instance.Q_f
        remaining_battery = battery_capacity if vehicle_type == 'electric' else None
        current_location = depot_index
        charged_once = False  # 标记是否已经充过电（仅对电动车）
        total_energy_consumed = 0  # 初始化总电量消耗
        current_time = 0  # 初始化当前时间（假设从0时刻开始）

        while customers:
            nearest_customer = None
            nearest_distance = float('inf')
            earliest_ready_time = float('inf')

            for customer in customers:
                distance = instance.d_ij[current_location, customer]
                travel_time = instance.t_ijk_e[current_location, customer] \
                    if vehicle_type == 'electric' else instance.t_ijk_f[current_location, customer]
                arrival_time = current_time + travel_time

                if arrival_time < instance.E_i[customer - 22]:  # 如果到达时间早于Ready Time
                    waiting_time = instance.E_i[customer - 22] - arrival_time
                    arrival_time = instance.E_i[customer - 22]  # 等待直到服务时间开始
                else:
                    waiting_time = 0

                if distance < nearest_distance and instance.q_i[customer - 22] <= remaining_capacity:
                    nearest_customer = customer
                    nearest_distance = distance
                    earliest_ready_time = arrival_time

            if nearest_customer is None:
                break

            # 计算到达客户的时间
            travel_time = instance.t_ijk_e[current_location, nearest_customer] if vehicle_type == 'electric' else \
                instance.t_ijk_f[current_location, nearest_customer]
            current_time += travel_time

            # 打印当前到达时间
            print(f"Vehicle {vehicle_type} arrives at customer {nearest_customer} at time {current_time:.2f}")

            # 电动车需要检查电量
            if vehicle_type == 'electric':
                battery_usage = instance.L_ijk_e[current_location, nearest_customer]
                battery_usage_back_to_depot = instance.L_ijk_e[nearest_customer, depot_index]

                # 检查是否在服务客户后电量足够返回配送中心
                if charged_once and remaining_battery - battery_usage < battery_usage_back_to_depot:
                    print(f"充电后服务客户 {nearest_customer} 的电量不足以返回配送中心，直接返回配送中心")
                    current_route.append(depot_index)
                    break

                # 检查电量是否足够
                if remaining_battery < battery_usage:
                    if charged_once or not charging_stations:
                        # 如果已经充过电或者没有可用的充电站且电量不足，则需要结束当前路径
                        break
                    # 找到最近的可用充电站
                    nearest_station = min(
                        charging_stations,
                        key=lambda station: instance.d_ij[current_location, station]
                    )
                    # 插入充电站到当前路径
                    current_route.append(nearest_station)
                    # 打印当前到达时间和等待时间
                    print(
                        f"Vehicle {vehicle_type} arrives at customer {nearest_customer} "
                        f"at time {arrival_time:.2f}, waiting for {waiting_time:.2f} time units.")

                    # 更新当前时间
                    current_time = earliest_ready_time + instance.locations[nearest_customer].service_time

                    # 假设在充电站将电池充满至B_star
                    remaining_battery = battery_capacity  # 电池充满
                    current_location = nearest_station
                    charged_once = True  # 标记已经充电一次
                    charging_stations.remove(nearest_station)  # 将充电站移除可用列表

            # 添加客户到当前路径
            current_route.append(nearest_customer)
            customers.remove(nearest_customer)
            remaining_capacity -= instance.q_i[nearest_customer - 22]

            if vehicle_type == 'electric':
                remaining_battery -= battery_usage  # 更新剩余电量
                total_energy_consumed += battery_usage  # 累积电量消耗

            current_location = nearest_customer

        # 添加返回配送中心并结束路径
        if vehicle_type == 'electric':
            battery_usage_back_to_depot = instance.L_ijk_e[current_location, depot_index]
            total_energy_consumed += battery_usage_back_to_depot  # 累积返回配送中心的电量消耗
            if remaining_battery >= battery_usage_back_to_depot:
                current_route.append(depot_index)
                current_time += instance.t_ijk_e[current_location, depot_index]
                print(f"Vehicle {vehicle_type} returns to depot at time {current_time:.2f}")
            else:
                if not charged_once:
                    # 找到最近的充电站并插入
                    nearest_station = min(
                        charging_stations,
                        key=lambda station: instance.d_ij[current_location, station]
                    )
                    current_route.append(nearest_station)
                    current_route.append(depot_index)
                    charging_stations.remove(nearest_station)  # 将充电站移除可用列表
                else:
                    print(f"警告：路径 {current_route} 的电量不足以返回配送中心")
        else:
            # 燃油车直接返回配送中心
            current_route.append(depot_index)
            current_time += instance.t_ijk_f[current_location, depot_index]
            print(f"Vehicle {vehicle_type} returns to depot at time {current_time:.2f}")

        routes.append((current_route, total_energy_consumed))

    return routes


def construct_initial_solution(instance: CustomEVRPInstance):
    """
    生成电动车和燃油车的初始解。
    """
    # 获取电动车和燃油车服务的客户
    electric_customers, fuel_customers = clustering(instance)

    # 将客户ID转换为从22开始的实际索引
    electric_customers = {customer + 21 for customer in electric_customers}  # 索引从1开始转换到从22开始
    fuel_customers = {customer + 21 for customer in fuel_customers}  # 同样转换

    # 生成电动车的路径
    electric_routes = nearest_neighbor_solution(instance, 'electric', electric_customers)

    # 生成燃油车的路径
    fuel_routes = nearest_neighbor_solution(instance, 'fuel', fuel_customers)

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
                                             for loc_id in route if loc_id in instance.type_to_indices['c']])
    if electric_customers_positions.size > 0:
        plt.scatter(electric_customers_positions[:, 0], electric_customers_positions[:, 1],
                    c='blue', label='Electric Vehicle Customers', marker='o', zorder=2)

    # 绘制燃油车客户
    fuel_customers_positions = np.array([(instance.locations[loc_id].x, instance.locations[loc_id].y)
                                         for route, _ in fuel_routes
                                         for loc_id in route if loc_id in instance.type_to_indices['c']])
    if fuel_customers_positions.size > 0:
        plt.scatter(fuel_customers_positions[:, 0], fuel_customers_positions[:, 1],
                    c='orange', label='Fuel Vehicle Customers', marker='x', zorder=2)

    # 绘制充电桩
    charging_stations_positions = np.array([(loc.x, loc.y) for loc in instance.locations if loc.type == 'f'])
    plt.scatter(charging_stations_positions[:, 0], charging_stations_positions[:, 1], c='green',
                label='Charging Stations', marker='^', zorder=4)

    # 使用不同颜色绘制路径
    colors = itertools.cycle(plt.cm.get_cmap('tab20').colors)

    # 绘制电动车路线
    for route, energy_consumed in electric_routes:
        route_positions = [instance.locations[loc_id].x for loc_id in route], \
            [instance.locations[loc_id].y for loc_id in route]
        color = next(colors)
        plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='--', color=color,
                 label=f'Electric Route: Energy Consumed {energy_consumed:.2f}', zorder=1)

    # 绘制燃油车路线
    for route, _ in fuel_routes:
        route_positions = [instance.locations[loc_id].x for loc_id in route], \
            [instance.locations[loc_id].y for loc_id in route]
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
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 生成电动车和燃油车的初始解
    electric_routes, fuel_routes = construct_initial_solution(instance)

    # 打印生成的路径及其电量消耗
    for i, (route, energy_consumed) in enumerate(electric_routes):
        print(f"Electric vehicle route {i + 1}: {route}, Energy consumed: {energy_consumed:.2f}")
    for i, (route, _) in enumerate(fuel_routes):
        print(f"Fuel vehicle route {i + 1}: {route}")
    # 绘制路径图
    plot_routes_with_stations(instance, electric_routes, fuel_routes,
                              "Electric and Fuel Vehicle Routes with Charging Stations")
    print("Distance matrix (d_ij):")
    print(instance.d_ij[:10, :10])  # 打印矩阵的前10行和前10列

    print("Energy consumption coefficients (L_ijk_e):")
    print(instance.L_ijk_e[:10, :10])  # 打印矩阵的前10行和前10列


if __name__ == "__main__":
    main()
