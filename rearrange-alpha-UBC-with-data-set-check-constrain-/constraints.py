def node_visit_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    M = instance.M

    max_index = min(len(instance.x), len(instance.locations))

    constraints.append(
        sum(instance.x[0, j, k] for j in C + R if j < max_index for k in K if k < max_index) ==
        sum(instance.x[j, instance.O_prime, k] for j in C + R if j < max_index for k in K if k < max_index)
    )

    for i in C:
        if i < max_index:
            constraints.append(
                sum(instance.x[i, j, k] for j in N if j != i and j < max_index for k in K if k < max_index) == 1
            )

    for j in C:
        if j < max_index:
            for k in K:
                if k < max_index:
                    constraints.append(
                        sum(instance.x[i, j, k] for i in N if i != j and i < max_index) ==
                        sum(instance.x[j, i, k] for i in N if i != j and i < max_index)
                    )

    for k in K_e:
        if k < max_index:
            constraints.append(
                sum(instance.x[i, j, k] for i in N if i < max_index for j in R if j < max_index) <= instance.sigma
            )

    return constraints

def load_balance_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    E = [(i, j) for i in N for j in N if i != j]
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    K_f = list(range(instance.k_e, instance.k_e + instance.k_f))
    M = instance.M
    Q_e = instance.Q_e
    Q_f = instance.Q_f
    q = instance.customer_demand

    max_index = min(len(instance.u), len(instance.locations))

    for (i, j) in E:
        if i < max_index and j < max_index:
            for k in K_e:
                if k < max_index:
                    constraints.append(instance.u[i, j, k] <= Q_e)
            for k in K_f:
                if k < max_index:
                    constraints.append(instance.u[i, j, k] <= Q_f)

    for j in C:
        if j < max_index and j < len(q):
            for k in K:
                if k < max_index:
                    constraints.append(
                        sum(instance.u[i, j, k] for i in N if i != j and i < max_index) -
                        sum(instance.u[j, i, k] for i in N if i != j and i < max_index) +
                        M * (1 - sum(instance.x[i, j, k] for i in N if i != j and i < max_index)) >= q[j]
                    )

    for k in K:
        if k < max_index:
            constraints.append(
                sum(instance.u[i, instance.O_prime, k] for i in C + R if i < max_index) == 0
            )

    return constraints

def time_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    M = instance.M
    t = instance.travel_time_matrix
    s = instance.service_time
    w = np.zeros((instance.N, len(K)))  # 确保w是二维数组
    E = [loc['ready_time'] for loc in instance.locations]
    L = [loc['due_date'] for loc in instance.locations]

    max_index = min(len(instance.x), len(instance.locations), len(t))

    for k in K:
        if k < len(instance.b):  # 添加边界检查
            constraints.append(instance.b[0, k] >= E[0])

    for j in C:
        if j < max_index:
            for k in K:
                if k < len(instance.b) and j < len(instance.a):  # 添加边界检查
                    constraints.append(
                        instance.b[0, k] + t[0, j] * instance.x[0, j, k] - M * (1 - instance.x[0, j, k]) <= instance.a[j, k]
                    )

    for i in C:
        if i < max_index:
            for j in N:
                if j != i and j < max_index:
                    for k in K:
                        if k < len(instance.a) and i < len(instance.a) and j < len(instance.a):  # 添加边界检查
                            constraints.append(
                                instance.a[i, k] + t[i, j] * instance.x[i, j, k] + s[i] + w[i, k] - M * (1 - instance.x[i, j, k]) <= instance.a[j, k]
                            )

    for i in N:
        if i < max_index:
            for k in K:
                if k < len(instance.b):  # 添加边界检查
                    constraints.append(
                        instance.b[i, k] + t[i, instance.O_prime] * instance.x[i, instance.O_prime, k] - M * (1 - instance.x[i, instance.O_prime, k]) <= L[0]
                    )

    for i in C:
        if i < max_index:
            for k in K:
                if k < len(instance.a):  # 添加边界检查
                    constraints.append(
                        instance.b[i, k] == instance.a[i, k] + s[i] + w[i, k]
                    )

    for i in R:
        if i < max_index:
            for k in K_e:
                if k < len(instance.b):  # 添加边界检查
                    constraints.append(
                        instance.b[i, k] == instance.a[i, k] + M
                    )

    for i in C:
        if i < max_index:
            for k in K_e:
                if k < len(instance.a):  # 添加边界检查
                    constraints.append(
                        E[i] <= instance.a[i, k] <= L[i]
                    )

    return constraints

def battery_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K_e = list(range(instance.k_e))
    M = instance.M
    L_ijk = instance.L_ijk
    B_star = instance.B_star
    soc_min = instance.soc_min

    for i in N:
        for j in N:
            if j != i:
                for k in K_e:
                    constraints.append(
                        instance.B[i, k] - L_ijk[i, j] * instance.x[i, j, k] + B_star * (1 - instance.x[i, j, k]) >= instance.B[j, k]
                    )

    for k in K_e:
        constraints.append(
            instance.B[0, k] == B_star
        )

    for i in N:
        for k in K_e:
            constraints.append(
                instance.B[i, k] >= B_star * soc_min
            )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.B[r, k] == B_star
            )

    return constraints

def variable_constraints(instance):
    constraints = []

    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))

    for (i, j) in [(i, j) for i in N for j in N if i != j]:
        for k in K:
            constraints.append(instance.u[i, j, k] >= 0)
            if k in range(instance.k_e):
                constraints.append(instance.f[i, j, k] >= 0)
            constraints.append(instance.x[i, j, k] == 0 or instance.x[i, j, k] == 1)

    return constraints
