def nearest_neighbor_solution(instance, vehicle_type, customers):
    """
    使用最近邻算法生成车辆的初始解，并且管理电量，在电量不足时加入充电站。
    每条路径最多只允许充一次电，每个充电站只能在路径上使用一次。
    """
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
        current_time = 0  # 初始化当前时间为0

        while customers:
            nearest_customer = None
            nearest_distance = float('inf')

            for customer in customers:
                distance = instance.d_ij[current_location, customer]
                if distance < nearest_distance and instance.q_i[customer - 22] <= remaining_capacity:
                    nearest_customer = customer
                    nearest_distance = distance

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


