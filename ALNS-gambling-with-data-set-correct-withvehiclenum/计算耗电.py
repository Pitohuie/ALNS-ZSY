# 参数设置
v_e = 11.11  # m/s
d = 10000  # 米
m_v_e = 2100  # 公斤
rho = 1.225  # kg/m³
A = 2.5  # m²
g = 9.81  # m/s²
c_d = 0.3
phi_d = 0.9
varphi_d = 0.85

# 计算 K_{ijk}^e
K_ijk_e = 0.5 * c_d * rho * A * v_e**3 + (m_v_e * g * c_d * v_e)

# 计算 L_{ijk}^e
L_ijk_e = phi_d * varphi_d * K_ijk_e * (d / v_e)

# 转换为千瓦时
energy_kwh = L_ijk_e * 2.77778e-7

print(f"能量消耗: {energy_kwh} kWh")
