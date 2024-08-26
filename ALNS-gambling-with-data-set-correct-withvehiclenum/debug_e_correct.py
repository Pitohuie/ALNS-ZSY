import numpy as np
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from a_read_instance import read_solomon_instance
from e_recorrect import construct_initial_solution_with_partial_charging
import matplotlib.pyplot as plt

def main():
    # 1. 读取实例文件并初始化
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 2. 生成电动车和燃油车的初始解
    electric_routes, fuel_routes = construct_initial_solution_with_partial_charging(instance)

    # 3. 打印生成的路径
    print("Electric vehicle routes:", electric_routes)
    print("Fuel vehicle routes:", fuel_routes)

    # 4. 绘制路径图
    plot_routes_with_stations(instance, electric_routes, fuel_routes, "Initial Solution with Partial Charging")

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
    colors = plt.cm.get_cmap('tab20').colors

    # 绘制电动车路线
    for i, route in enumerate(electric_routes):
        route_positions = [instance.locations[loc_id].x for loc_id in route], [instance.locations[loc_id].y for loc_id in route]
        plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='--', color=colors[i % len(colors)],
                 label=f'Electric Vehicle Route {i+1}')

    # 绘制燃油车路线
    for i, route in enumerate(fuel_routes):
        route_positions = [instance.locations[loc_id].x for loc_id in route], [instance.locations[loc_id].y for loc_id in route]
        plt.plot(route_positions[0], route_positions[1], marker='o', linestyle='-', color=colors[(i + len(electric_routes)) % len(colors)],
                 label=f'Fuel Vehicle Route {i+1}')

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
