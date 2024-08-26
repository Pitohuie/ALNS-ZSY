from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance,Location
from a_read_instance import read_solomon_instance
from e_initial_solution_with_clustering import clustering,greedy_initial_solution_with_partial_charging, greedy_initial_solution_for_fuel
import matplotlib.pyplot as plt

def main():
    # 1. 读取实例文件并初始化
    file_path = "c101_21.txt"  # 替换为您的实际文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    locations = [Location(**loc) for loc in locations_data]
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 2. 分别生成电动车和燃油车的初始解
    electric_customers, fuel_customers = clustering(instance)
    electric_routes = greedy_initial_solution_with_partial_charging(instance, 'electric', electric_customers)
    fuel_routes = greedy_initial_solution_for_fuel(instance, 'fuel', fuel_customers)

    # 3. 打印生成的路径
    print("Electric Vehicle Routes:")
    for route in electric_routes:
        print(route)

    print("\nFuel Vehicle Routes:")
    for route in fuel_routes:
        print(route)

    # 4. 可视化路径
    plot_routes_with_charging(instance, electric_routes, fuel_routes)


def plot_routes_with_charging(instance, electric_routes, fuel_routes):
    """
    可视化电动车和燃油车的路径，包括充电桩。
    """
    customer_positions = instance.customer_positions
    depot_position = instance.depot_position
    charging_stations = [customer_positions[i - 1] for i in instance.R]

    plt.figure(figsize=(10, 6))

    # 绘制电动车路径
    for route in electric_routes:
        route_positions = [depot_position] + [customer_positions[i - 1] for i in route if i != 0] + [depot_position]
        plt.plot([pos[0] for pos in route_positions], [pos[1] for pos in route_positions], 'b-o', label='Electric Vehicle Route')

    # 绘制燃油车路径
    for route in fuel_routes:
        route_positions = [depot_position] + [customer_positions[i - 1] for i in route if i != 0] + [depot_position]
        plt.plot([pos[0] for pos in route_positions], [pos[1] for pos in route_positions], 'r-x', label='Fuel Vehicle Route')

    # 绘制仓库位置
    plt.scatter(depot_position[0], depot_position[1], color='green', label='Depot', marker='s', s=100)

    # 绘制充电桩位置
    for station in charging_stations:
        plt.scatter(station[0], station[1], color='purple', label='Charging Station', marker='^', s=100)

    plt.title('Electric and Fuel Vehicle Routes with Charging Stations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
