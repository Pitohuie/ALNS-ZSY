from typing import List
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, CustomerLocation, ChargingStationLocation, DepotLocation

class Constraints:
    def __init__(self, instance: CustomEVRPInstance):
        self.instance = instance

        # 初始化客户、充电站和配送中心的索引集合
        self.C = [idx for idx, loc in enumerate(instance.locations) if isinstance(loc, CustomerLocation)]
        self.R = [idx for idx, loc in enumerate(instance.locations) if isinstance(loc, ChargingStationLocation)]
        self.depot_index = next(idx for idx, loc in enumerate(instance.locations) if isinstance(loc, DepotLocation))

        # 初始化所有节点的索引集合
        self.N = list(range(instance.N))

        # 常量与参数
        self.M = instance.M
        self.Q_e = instance.Q_e
        self.Q_f = instance.Q_f
        self.q = instance.q_i
        self.t_e = instance.t_ijk_e
        self.t_f = instance.t_ijk_f
        self.s = instance.q_i
        self.w = instance.w_ik
        self.E = instance.E_i
        self.L = instance.L_i
        self.B_star = instance.B_star
        self.L_ijk_e = instance.L_ijk_e
        self.soc_min = instance.soc_min
        self.e = instance.e

        # 初始化车辆数量相关的属性
        self.update_vehicle_related_data()

    def update_vehicle_related_data(self):
        """更新与车辆数量相关的内部数据。"""
        self.K = list(range(self.instance.K_e + self.instance.K_f))
        self.K_e = list(range(self.instance.K_e))
        self.K_f = list(range(self.instance.K_e, self.instance.K_e + self.instance.K_f))

    def check_node_visit(self, route) -> bool:
        """检查节点访问约束。"""
        # 检查路径是否从配送中心开始并回到配送中心
        if route[0] != self.depot_index or route[-1] != self.depot_index:
            return False

        # 检查所有客户是否都被访问且路径无重复节点
        if len(set(route)) != len(route):
            return False
        if not all(customer in route for customer in self.C):
            return False

        return True

    def check_load_balance(self, route, vehicle_type: str) -> bool:
        """检查负载平衡约束。"""
        total_demand = sum(self.q[node] for node in route if node in self.C)
        if vehicle_type == 'electric' and total_demand > self.Q_e:
            return False
        if vehicle_type == 'fuel' and total_demand > self.Q_f:
            return False
        return True

    def check_time_window(self, route) -> bool:
        """检查时间窗约束。"""
        current_time = 0

        # 从配送中心开始
        current_time = max(self.E[self.depot_index], current_time)  # 等待配送中心的准备时间
        current_time += self.s[self.depot_index]  # 考虑配送中心的服务时间

        for i in range(1, len(route)):
            current_node = route[i - 1]
            next_node = route[i]

            # 旅行时间累加
            travel_time = self.t_e[current_node, next_node]
            current_time += travel_time

            # 检查是否满足下一节点的时间窗
            ready_time = self.E[next_node]
            due_date = self.L[next_node]

            if current_time < ready_time:
                # 当前时间早于时间窗的开始时间，等待直到时间窗开始
                current_time = ready_time

            # 检查是否违反时间窗约束
            if current_time > due_date:
                return False

            # 增加服务时间
            service_time = self.s[next_node]
            current_time += service_time

        return True

    def check_battery(self, route) -> bool:
        """检查电池容量约束，仅允许一次充电。"""
        remaining_battery = self.B_star
        charged_once = False  # 记录是否已充电一次

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            battery_usage = self.L_ijk_e[current_node, next_node]

            if current_node in self.R and not charged_once:  # 当前节点是充电站且未充电
                # 判断是否需要全充电
                if self.needs_full_charge(route[i:], remaining_battery):
                    remaining_battery = self.B_star  # 全充电
                else:
                    # 计算到达下一充电站或路径终点所需的电量
                    next_station_index = next((j for j in range(i + 1, len(route)) if route[j] in self.R),
                                              len(route) - 1)
                    required_battery_to_next_station = sum(
                        self.L_ijk_e[route[k], route[k + 1]] for k in range(i, next_station_index))

                    # 根据需求部分充电，避免过度充电
                    remaining_battery += self.calculate_partial_charge(current_node,
                                                                       required_battery_to_next_station - remaining_battery)
                    remaining_battery = min(remaining_battery, self.B_star)  # 防止超过电池容量

                charged_once = True  # 标记已经充电一次

            # 检查电池是否足够支持到下一个节点
            if remaining_battery < battery_usage:
                return False

            remaining_battery -= battery_usage

        # 如果未经过充电站或经过多个充电站并试图充电多次，视为违反约束
        if not charged_once or sum(1 for node in route if node in self.R) > 1:
            return False

        return remaining_battery >= self.B_star * self.soc_min

    def calculate_partial_charge(self, station_id, needed_charge):
        """根据等待时间和需求量计算部分充电量。"""
        wait_time = self.instance.w_ik[station_id, 0]
        charging_rate = self.instance.vehicles['inverse_refueling_rate']  # 从字典中获取充电速率
        partial_charge = charging_rate * wait_time
        # 仅充电到需要的量，避免过度充电
        return min(partial_charge, needed_charge)

    def needs_full_charge(self, remaining_route, current_battery) -> bool:
        """判断是否需要全充电"""
        estimated_usage = sum(
            self.L_ijk_e[remaining_route[i], remaining_route[i + 1]] for i in range(len(remaining_route) - 1))
        # 如果预计使用量接近当前电量，或者接下来很少有充电站，选择全充电
        return estimated_usage >= current_battery * 0.8 or not any(node in self.R for node in remaining_route[1:])

    def node_visit_constraints(self) -> List:
        constraints = []

        for k in self.K:
            constraints.append(
                sum(self.instance.x_ijk[self.depot_index, j, k] for j in
                    self.C + self.R) ==
                sum(self.instance.x_ijk[j, self.instance.O_prime, k] for j in self.C + self.R)
            )

        for i in self.C:
            constraints.append(
                sum(self.instance.x_ijk[i, j, k] for j in self.N if j != i for k in self.K) == 1
            )

        for j in self.C:
            for k in self.K:
                constraints.append(
                    sum(self.instance.x_ijk[i, j, k] for i in self.N if i != j) ==
                    sum(self.instance.x_ijk[j, i, k] for i in self.N if i != j)
                )

        for k in self.K_e:
            constraints.append(
                sum(self.instance.x_ijk[i, j, k] for i in self.N for j in self.R) <= 1
            )

        return constraints

    def load_balance_constraints(self) -> List:
        constraints = []

        E = [(i, j) for i in self.N for j in self.N if i != j]

        for (i, j) in E:
            for k in self.K_e:
                constraints.append(self.instance.u_ijk[i, j, k] <= self.Q_e)
            for k in self.K_f:
                constraints.append(self.instance.u_ijk[i, j, k] <= self.Q_f)

        for j in self.C:
            for k in self.K:
                constraints.append(
                    sum(self.instance.u_ijk[i, j, k] for i in self.N if i != j) -
                    sum(self.instance.u_ijk[j, i, k] for i in self.N if i != j) +
                    self.M * (1 - sum(self.instance.x_ijk[i, j, k] for i in self.N if i != j)) >= self.q[j]
                )

        for k in self.K:
            constraints.append(
                sum(self.instance.u_ijk[i, self.instance.O_prime, k] for i in self.C + self.R) == 0
            )

        return constraints

    def time_constraints(self) -> List:
        constraints = []

        for k in self.K:
            # Ensure vehicles leave the depot no earlier than the open time
            constraints.append(
                self.instance.b_ik[self.depot_index, k] >= self.E[self.depot_index])

        for j in self.C:
            for k in self.K:
                constraints.append(
                    self.instance.b_ik[self.depot_index, k] +
                    self.t_e[self.depot_index, j] *
                    self.instance.x_ijk[self.depot_index, j, k] -
                    self.M * (1 - self.instance.x_ijk[self.depot_index, j, k]) <=
                    self.instance.a_ik[j, k]
                )

        for i in self.C:
            for j in self.N:
                if i != j:
                    for k in self.K:
                        constraints.append(
                            self.instance.a_ik[i, k] + self.t_e[i, j] * self.instance.x_ijk[i, j, k] + self.s[i] +
                            self.w[i, k] - self.M * (1 - self.instance.x_ijk[i, j, k]) <= self.instance.a_ik[j, k]
                        )

        for i in self.R:
            for j in self.N:
                if i != j:
                    for k in self.K_e:
                        constraints.append(
                            self.instance.a_ik[i, k] + self.t_e[i, j] * self.instance.x_ijk[i, j, k] + self.s[i] +
                            self.w[i, k] - self.M * (1 - self.instance.x_ijk[i, j, k]) <= self.instance.a_ik[j, k]
                        )

        for i in self.N:
            for k in self.K:
                constraints.append(
                    self.instance.b_ik[i, k] +
                    self.t_e[i, self.instance.O_prime] *
                    self.instance.x_ijk[i, self.instance.O_prime, k] -
                    self.M * (1 - self.instance.x_ijk[i, self.instance.O_prime, k]) <=
                    self.L[self.depot_index]
                )

        for i in self.C:
            for k in self.K:
                constraints.append(
                    self.instance.a_ik[i, k] >= self.E[i]
                )
                constraints.append(
                    self.instance.a_ik[i, k] + self.w[i, k] <= self.L[i]
                )
                constraints.append(
                    self.instance.w_ik[i, k] >= self.E[i] - self.instance.a_ik[i, k]
                )

        for i in self.N:
            for j in self.N:
                if i != j:
                    for k in self.K_e:
                        constraints.append(
                            self.instance.B_ik1[i, k] - self.L_ijk_e[i, j] * self.instance.x_ijk[
                                i, j, k] + self.B_star * (1 - self.instance.x_ijk[i, j, k]) >= self.instance.B_ik1[j, k]
                        )

        for k in self.K_e:
            constraints.append(
                self.instance.B_ik1[self.depot_index, k] == self.B_star
            )

        for i in self.N:
            for k in self.K_e:
                constraints.append(
                    self.instance.B_ik1[i, k] >= self.B_star * self.soc_min
                )

        for r in self.R:
            for k in self.K_e:
                constraints.append(
                    self.instance.B_ik1[r, k] == self.B_star
                )

        for (i, j) in [(i, j) for i in self.N for j in self.N if i != j]:
            for k in self.K:
                constraints.append(self.instance.u_ijk[i, j, k] >= 0)
                if k in self.K_e:
                    constraints.append(self.instance.f_ijk[i, j, k] >= 0)
                constraints.append(self.instance.x_ijk[i, j, k] == 0 or self.instance.x_ijk[i, j, k] == 1)

        return constraints

    def optimize_charging_stations(self, route):
        """优化路径中充电站的使用"""
        for i in range(1, len(route) - 1):
            if route[i] in self.R:
                # 如果在下一个充电站前电量足够且当前充电站等待时间长，考虑跳过这个充电站
                next_station_index = next((j for j in range(i + 1, len(route)) if route[j] in self.R), None)
                if next_station_index:
                    required_battery_to_next_station = sum(
                        self.L_ijk_e[route[k], route[k + 1]] for k in range(i, next_station_index))
                    if self.B_star - required_battery_to_next_station >= self.soc_min * self.B_star:
                        continue
                # 否则继续在当前站点充电
                remaining_battery = self.calculate_partial_charge(route[i])
