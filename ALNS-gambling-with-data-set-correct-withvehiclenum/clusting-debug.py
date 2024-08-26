from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from e_initial_solution_with_clustering import clustering  # 确保导入了修改后的聚类函数
from e_initial_solution_with_clustering import calculate_scores
from a_read_instance import read_solomon_instance
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 读取实例文件并初始化
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 2. 计算评分并运行聚类算法
    electric_customers, fuel_customers = clustering(instance, 0.5)

    # 3. 转换为 0-based 索引
    electric_customers = np.array(electric_customers) - 1
    fuel_customers = np.array(fuel_customers) - 1

    # 4. 绘制聚类结果的散点图
    plot_clustering(instance, electric_customers, fuel_customers)

def plot_clustering(instance, E, C):
    """
    绘制聚类后的客户位置分布，包括充电站。

    E: 电动车服务的客户列表
    C: 燃油车服务的客户列表
    """
    # 提取客户和仓库的位置
    customer_positions = instance.customer_positions
    depot_position = instance.depot_position

    # 提取充电站位置
    charging_station_positions = np.array([(loc.x, loc.y) for loc in instance.locations if loc.type == 'f'])

    # 绘制电动车客户
    if E.size > 0:
        plt.scatter(customer_positions[E, 0], customer_positions[E, 1], color='blue',
                    label='Electric Vehicle Customers', marker='o')

    # 绘制燃油车客户
    if C.size > 0:
        plt.scatter(customer_positions[C, 0], customer_positions[C, 1], color='red', label='Fuel Vehicle Customers',
                    marker='x')

    # 绘制充电站
    if charging_station_positions.size > 0:
        plt.scatter(charging_station_positions[:, 0], charging_station_positions[:, 1], color='purple',
                    label='Charging Stations', marker='^')

    # 绘制仓库位置
    plt.scatter(depot_position[0], depot_position[1], color='green', label='Depot', marker='s', s=100)

    # 添加标题和标签
    plt.title('Customer Clustering for Electric and Fuel Vehicles with Charging Stations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
