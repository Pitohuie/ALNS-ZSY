import matplotlib.pyplot as plt
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from e_initial_solution_with_clustering import construct_initial_solution_with_partial_charging
from a_read_instance import read_solomon_instance

# Step 1: 读取Solomon实例数据
file_path = "c101_21.txt"  # 确保文件路径正确
locations_data, vehicles_data = read_solomon_instance(file_path)

# Step 2: 创建Location对象列表，并创建CustomEVRPInstance对象
locations = [Location(**loc) for loc in locations_data]  # 使用直接导入的Location类
instance = CustomEVRPInstance(locations, vehicles_data)

# Step 3: 使用聚类和贪婪算法生成初始解
electric_routes, fuel_routes = construct_initial_solution_with_partial_charging(instance)

# Step 4: 绘制路径
def plot_routes(instance, routes, title, color):
    plt.figure(figsize=(10, 8))
    depot_x, depot_y = instance.depot_position
    plt.plot(depot_x, depot_y, 'rs', markersize=10, label='Depot')  # Plot depot

    for route in routes:
        route_x = [instance.locations[node].x for node in route]
        route_y = [instance.locations[node].y for node in route]
        plt.plot(route_x, route_y, marker='o', color=color, label='Route')
        plt.scatter(route_x, route_y, color=color)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制电动车路径
plot_routes(instance, electric_routes, 'Electric Vehicle Routes', 'blue')

# 绘制燃油车路径
plot_routes(instance, fuel_routes, 'Fuel Vehicle Routes', 'green')
