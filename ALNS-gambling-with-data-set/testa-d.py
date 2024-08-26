from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from c_constraints import Constraints
from e_initial_solution_with_clustering import construct_initial_solution_with_partial_charging
from a_read_instance import read_solomon_instance

def main():
    # 1. 读取Solomon实例文件
    file_path = "c101_21.txt"  # 替换为你的Solomon实例文件的路径
    try:
        locations_data, vehicles_data = read_solomon_instance(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return

    # 2. 创建Location对象列表
    locations = [Location(**loc) for loc in locations_data]

    # 3. 创建CustomEVRPInstance对象
    instance = CustomEVRPInstance(locations, vehicles_data)

    # 4. 创建Constraints对象（用于检查约束条件）
    constraints_checker = Constraints(instance)

    # 5. 生成电动车和燃油车的初始解
    electric_routes, fuel_routes = construct_initial_solution_with_partial_charging(instance)

    # 6. 打印生成的初始解结果
    print("电动车初始路径:")
    for route in electric_routes:
        print(route)

    print("\n燃油车初始路径:")
    for route in fuel_routes:
        print(route)

    # 7. 验证约束条件（可选）
    print("\n验证约束条件...")
    for route in electric_routes + fuel_routes:
        if not constraints_checker.check_all_constraints(route):
            print(f"路径 {route} 未能满足所有约束条件！")
        else:
            print(f"路径 {route} 满足所有约束条件。")

if __name__ == "__main__":
    main()
