from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from e_initial_solution_with_clustering import clustering
from a_read_instance import read_solomon_instance
import numpy as np
import matplotlib.pyplot as plt
import itertools

def nearest_neighbor_solution(instance, vehicle_type, customers):
    """
    使用最近邻算法生成车辆的初始解。
    """
    routes = []

    while customers:
        current_route = [instance.depot_index]  # 路线从配送中心开始
        remaining_capacity = instance.Q_e if vehicle_type == 'electric' else instance.Q_f
        current_location = instance.depot_index

        while customers:
            nearest_customer = None
            nearest_distance = float('inf')

            for customer in customers:
                adjusted_customer_index = customer - 22  # 因为客户ID从22开始，所以需要减去22得到正确的索引
                distance = instance.d_ij[current_location, customer]
                if distance < nearest_distance and instance.q_i[adjusted_customer_index] <= remaining_capacity:
                    nearest_customer = customer
                    nearest_distance = distance

            if nearest_customer is None:
                break

            # 添加客户到当前路径
            current_route.append(nearest_customer)
            customers.remove(nearest_customer)
            adjusted_customer_index = nearest_customer - 22
            remaining_capacity -= instance.q_i[adjusted_customer_index]
            current_location = nearest_customer

        # 添加返回配送中心并结束路径
        current_route.append(instance.depot_index)
        routes.append(current_route)

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
    绘制给定路径的路线图，包括充电桩位置。
    """
    plt.figure(figsize=(10, 8))

    # 绘制所有客户和站点的位置
    customer_positions = np.array([(loc.x, loc.y) for loc in instance.locations if loc.type == 'c'])
    depot_position = np.array([instance.O.x, instance.O.y])
    charging_stations_positions = np.array([(loc.x, loc.y) for loc in instance.locations if loc.type == 'f'])

    # 绘制客户、配送中心和充电桩
    plt.scatter(customer_positions[:, 0], customer_positions[:, 1], c='blue', label='Customers')
    plt.scatter(charging_stations_positions[:, 0], charging_stations_positions[:, 1], c='green',
                label='Charging Stations', marker='^')
    plt.scatter(depot_position[0], depot_position[1], c='red', label='Depot', marker='s')

    # 使用不同颜色绘制路径
    colors = itertools.cycle(plt.cm.get_cmap('tab20').colors)

    # 绘制电动车路线
    for route in electric_routes:
        route_positions = [instance.locations[loc_id].x for loc_id in route], [instance.locations[loc_id].y for loc_id in route]
        color = next(colors)
        plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='--', color=color,
                 label='Electric Vehicle Routes')

    # 绘制燃油车路线
    for route in fuel_routes:
        route_positions = [instance.locations[loc_id].x for loc_id in route], [instance.locations[loc_id].y for loc_id in route]
        color = next(colors)
        plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='-', color=color,
                 label='Fuel Vehicle Routes')

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

    # 打印生成的路径
    print("Electric vehicle routes:", electric_routes)
    print("Fuel vehicle routes:", fuel_routes)

    # 绘制路径图
    plot_routes_with_stations(instance, electric_routes, fuel_routes,
                              "Electric and Fuel Vehicle Routes with Charging Stations")

if __name__ == "__main__":
    main()
