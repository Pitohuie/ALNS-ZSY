from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from a_read_instance import read_solomon_instance  # 读取实例文件的函数
from e_initial_solution_with_clustering import clustering  # 调用聚类算法
from c_constraints import Constraints  # 假设这是您的约束文件
import matplotlib.pyplot as plt
import numpy as np


def plot_routes(instance, electric_routes, fuel_routes):
    """
    绘制电动车和燃油车的路径。
    """
    # 提取客户和仓库的位置
    customer_positions = instance.customer_positions
    depot_position = instance.depot_position

    plt.figure(figsize=(10, 8))

    # 绘制电动车的路径
    for route in electric_routes:
        route_positions = customer_positions[np.array(route) - 1]
        plt.plot(route_positions[:, 0], route_positions[:, 1], 'bo-', label='Electric Vehicle Route')
        plt.plot([depot_position[0], route_positions[0, 0]], [depot_position[1], route_positions[0, 1]], 'bo--')
        plt.plot([route_positions[-1, 0], depot_position[0]], [route_positions[-1, 1], depot_position[1]], 'bo--')

    # 绘制燃油车的路径
    for route in fuel_routes:
        route_positions = customer_positions[np.array(route) - 1]
        plt.plot(route_positions[:, 0], route_positions[:, 1], 'rx-', label='Fuel Vehicle Route')
        plt.plot([depot_position[0], route_positions[0, 0]], [depot_position[1], route_positions[0, 1]], 'rx--')
        plt.plot([route_positions[-1, 0], depot_position[0]], [route_positions[-1, 1], depot_position[1]], 'rx--')

    # 绘制仓库位置
    plt.plot(depot_position[0], depot_position[1], 'gs', label='Depot', markersize=10)

    plt.title('Vehicle Routes for Electric and Fuel Vehicles')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def main():
    # 1. 读取实例文件并初始化
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 2. 初始化约束条件检查器
    constraints_checker = Constraints(instance)

    # 3. 调用聚类算法
    electric_customers, fuel_customers = clustering(instance, lambda_param=0.5)

    # 4. 检查约束条件
    if constraints_checker.check_node_visit(electric_customers) and constraints_checker.check_load_balance(
            electric_customers, 'electric'):
        print("Electric Vehicle Customer Set (E) satisfies constraints.")
    else:
        print("Electric Vehicle Customer Set (E) violates constraints.")

    if constraints_checker.check_node_visit(fuel_customers) and constraints_checker.check_load_balance(fuel_customers,
                                                                                                       'fuel'):
        print("Fuel Vehicle Customer Set (C) satisfies constraints.")
    else:
        print("Fuel Vehicle Customer Set (C) violates constraints.")

    # 5. 生成路径并绘制
    electric_routes = [electric_customers]  # 假设将所有电动车客户作为一条路径
    fuel_routes = [fuel_customers]  # 假设将所有燃油车客户作为一条路径
    plot_routes(instance, electric_routes, fuel_routes)


if __name__ == "__main__":
    main()
