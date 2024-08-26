import numpy as np

def define_constraints(instance):
    constraints = []

    # 基本参数和集合
    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    K_f = list(range(instance.k_e, self.k_e + instance.k_f))
    M = instance.M

    # 节点访问约束
    constraints.append(
        sum(instance.x[0, j, k] for j in C + R for k in K) == sum(instance.x[j, instance.O_prime, k] for j in C + R for k in K)
    )

    for i in C:
        constraints.append(
            sum(instance.x[i, j, k] for j in N if j != i for k in K) == 1
        )

    for j in C:
        for k in K:
            constraints.append(
                sum(instance.x[i, j, k] for i in N if i != j) == sum(instance.x[j, i, k] for i in N if i != j)
            )

    for r in R:
        for k in K_e:
            constraints.append(
                sum(instance.x[i, r, k] for i in N if i != r) <= 1
            )
            constraints.append(
                sum(instance.x[r, j, k] for j in N if j != r) <= 1
            )

    # 载重约束和货流平衡约束
    for (i, j) in [(i, j) for i in N for j in N if i != j]:
        for k in K:
            if k in K_e:
                constraints.append(instance.u[i, j, k] <= instance.Q_e)
            if k in K_f:
                constraints.append(instance.u[i, j, k] <= instance.Q_f)

    for j in C:
        for k in K:
            constraints.append(
                sum(instance.u[i, j, k] for i in N if i != j) - sum(instance.u[j, i, k] for i in N if i != j) + M * (1 - sum(instance.x[i, j, k] for i in N if i != j)) >= instance.customer_demand[j]
            )

    for k in K:
        constraints.append(
            sum(instance.u[i, instance.O_prime, k] for i in C + R) == 0
        )

    # 时间约束
    for k in K:
        constraints.append(instance.b[0, k] >= instance.time_window_start[0])

    for j in C:
        for k in K:
            constraints.append(
                instance.b[0, k] + instance.travel_time_matrix[0, j] * instance.x[0, j, k] - M * (1 - instance.x[0, j, k]) <= instance.a[j, k]
            )

    for i in C:
        for j in N:
            if j != i:
                for k in K:
                    constraints.append(
                        instance.a[i, k] + instance.travel_time_matrix[i, j] * instance.x[i, j, k] + instance.service_time[i] + instance.w[i, k] - M * (1 - instance.x[i, j, k]) <= instance.a[j, k]
                    )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.b[r, k] >= instance.a[r, k] + instance.B_star / instance.e
            )

    for i in N:
        for k in K:
            constraints.append(
                instance.b[i, k] + instance.travel_time_matrix[i, instance.O_prime] * instance.x[i, instance.O_prime, k] - M * (1 - instance.x[i, instance.O_prime, k]) <= instance.time_window_end[0]
            )

    for i in C:
        for k in K:
            constraints.append(
                instance.b[i, k] == instance.a[i, k] + instance.service_time[i] + instance.w[i, k]
            )

    for i in R:
        for k in K_e:
            constraints.append(
                instance.b[i, k] == instance.a[i, k] + instance.B_star / instance.e
            )

    for i in C:
        for k in K_e:
            constraints.append(
                instance.time_window_start[i] <= instance.a[i, k] <= instance.time_window_end[i]
            )

    # 电量约束
    for i in N:
        for j in N:
            if j != i:
                for k in K_e:
                    constraints.append(
                        instance.B[i, k] - instance.L_ijk[i, j] * instance.x[i, j, k] + instance.B_star * (1 - instance.x[i, j, k]) >= instance.B[j, k]
                    )

    for k in K_e:
        constraints.append(
            instance.B[0, k] == instance.B_star
        )

    for i in N:
        for k in K_e:
            constraints.append(
                instance.B[i, k] >= instance.B_star * instance.soc_min
            )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.B[r, k] == instance.B_star
            )

    # 变量取值约束
    for (i, j) in [(i, j) for i in N for j in N if i != j]:
        for k in K:
            constraints.append(instance.u[i, j, k] >= 0)
            if k in K_e:
                constraints.append(instance.f[i, j, k] >= 0)
            constraints.append(instance.x[i, j, k] == 0 or instance.x[i, j, k] == 1)

    return constraints
