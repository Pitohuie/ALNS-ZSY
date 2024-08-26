from typing import List
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance, Location
from a_read_instance import read_solomon_instance

class Constraints:
    def __init__(self, instance: CustomEVRPInstance):
        self.instance = instance

        self.C = [instance.location_id_to_index[loc.id] for loc in instance.locations if loc.type == 'c']
        self.R = [instance.location_id_to_index[loc.id] for loc in instance.locations if loc.type == 'f']
        self.N = list(range(instance.N))
        self.K = list(range(instance.K_e + instance.K_f))
        self.K_e = list(range(instance.K_e))
        self.K_f = list(range(instance.K_e, instance.K_e + instance.K_f))
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

    def node_visit_constraints(self) -> List:
        constraints = []

        for k in self.K:
            constraints.append(
                sum(self.instance.x_ijk[self.instance.location_id_to_index[self.instance.O.id], j, k] for j in self.C + self.R) ==
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
            constraints.append(self.instance.b_ik[self.instance.location_id_to_index[self.instance.O.id], k] >= self.E[0])

        for j in self.C:
            for k in self.K:
                constraints.append(
                    self.instance.b_ik[self.instance.location_id_to_index[self.instance.O.id], k] + self.t_e[self.instance.location_id_to_index[self.instance.O.id], j] * self.instance.x_ijk[self.instance.location_id_to_index[self.instance.O.id], j, k] - self.M * (1 - self.instance.x_ijk[self.instance.location_id_to_index[self.instance.O.id], j, k]) <= self.instance.a_ik[j, k]
                )

        for i in self.C:
            for j in self.N:
                if i != j:
                    for k in self.K:
                        constraints.append(
                            self.instance.a_ik[i, k] + self.t_e[i, j] * self.instance.x_ijk[i, j, k] + self.s[i] + self.w[i, k] - self.M * (1 - self.instance.x_ijk[i, j, k]) <= self.instance.a_ik[j, k]
                        )

        for i in self.R:
            for j in self.N:
                if i != j:
                    for k in self.K_e:
                        constraints.append(
                            self.instance.a_ik[i, k] + self.t_e[i, j] * self.instance.x_ijk[i, j, k] + self.s[i] + self.w[i, k] - self.M * (1 - self.instance.x_ijk[i, j, k]) <= self.instance.a_ik[j, k]
                        )

        for i in self.N:
            for k in self.K:
                constraints.append(
                    self.instance.b_ik[i, k] + self.t_e[i, self.instance.O_prime] * self.instance.x_ijk[i, self.instance.O_prime, k] - self.M * (1 - self.instance.x_ijk[i, self.instance.O_prime, k]) <= self.L[0]
                )

        for i in self.C:
            for k in self.K:
                constraints.append(
                    self.instance.b_ik[i, k] == self.instance.a_ik[i, k] + self.s[i] + self.w[i, k]
                )

        for i in self.R:
            for k in self.K_e:
                constraints.append(
                    self.instance.b_ik[i, k] == self.instance.a_ik[i, k] + self.M
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

        return constraints

    def battery_constraints(self) -> List:
        constraints = []

        for i in self.N:
            for j in self.N:
                if i != j:
                    for k in self.K_e:
                        constraints.append(
                            self.instance.B_ik1[i, k] - self.L_ijk_e[i, j] * self.instance.x_ijk[i, j, k] + self.B_star * (1 - self.instance.x_ijk[i, j, k]) >= self.instance.B_ik1[j, k]
                        )

        for k in self.K_e:
            constraints.append(
                self.instance.B_ik1[self.instance.location_id_to_index[self.instance.O.id], k] == self.B_star
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

        return constraints

    def variable_constraints(self) -> List:
        constraints = []

        for (i, j) in [(i, j) for i in self.N for j in self.N if i != j]:
            for k in self.K:
                constraints.append(self.instance.u_ijk[i, j, k] >= 0)
                if k in self.K_e:
                    constraints.append(self.instance.f_ijk[i, j, k] >= 0)
                constraints.append(self.instance.x_ijk[i, j, k] == 0 or self.instance.x_ijk[i, j, k] == 1)

        return constraints

if __name__ == "__main__":
    import os

    # 调试当前工作目录和文件列表
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir())

    # 使用绝对路径
    file_path = "D:/2024/ZSY-ALNS/pythonProject1/ALNS-gambling-with-data-set/c101_21.txt"  # 替换为你的文件绝对路径

    try:
        locations_data, vehicles_data = read_solomon_instance(file_path)
        locations = [Location(**loc) for loc in locations_data]
        instance = CustomEVRPInstance(locations, vehicles_data)
        constraints_instance = Constraints(instance)

        # 获取所有约束
        node_constraints = constraints_instance.node_visit_constraints()
        load_balance_constraints = constraints_instance.load_balance_constraints()
        time_constraints = constraints_instance.time_constraints()
        battery_constraints = constraints_instance.battery_constraints()
        variable_constraints = constraints_instance.variable_constraints()

        # 输出所有约束（用于调试）
        print("Node Visit Constraints:", node_constraints)
        print("Load Balance Constraints:", load_balance_constraints)
        print("Time Constraints:", time_constraints)
        print("Battery Constraints:", battery_constraints)
        print("Variable Constraints:", variable_constraints)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
