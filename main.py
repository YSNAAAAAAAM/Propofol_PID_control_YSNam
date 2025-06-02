# main.py
# 치과마취의 이해1 과제
# Yoon-Seo Nam, 2025-06-02

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

import Propofol_PK as pk
import Propofol_PD as pd

# PharmacoDynamics of propofol
age = 40      # years
wgt = 70.0    # kg
hgt = 170.0   # cm
m1f2 = 1.0    #
pma = 10.5    # weeks
tech = 1      # 1: arterial
a1v2 = 1      # 1: arterial
theta = np.array([
    1.83786, 3.23873, 5.60880, 0.581983, 0.559672, 0.103046, 0.191307,
    3.74422, 2.2033, -0.015633, -0.002857, 3.51313, -0.0138166, 4.22357,
    0.742043, 0.265642, 0.349885, -0.384927
])
ETA = np.zeros(7)

params_pk = pk.pk_parameters(age, wgt, hgt, m1f2, pma, tech, a1v2, theta, ETA)

# PharmacoKinetics of propofol
BIS_base = 92.9824
ce50 = 0.0

params = {
    'E50': np.exp(1.12475 + (-0.0063) * (age - 35)),  # θ1 + θ7*(AGE-35)
    #'KE0': np.exp(2.156050e-01) * (wgt / 70.0) ** (-0.25),  # venous assumed (A1V2=2)
    'KE0': np.exp(-1.92162) * (wgt / 70) ** (-0.25),  # arterial assumed (A1V2=1)
    'EMAX': 92.9824,
    'GAM': np.exp(0.387721),  # θ4
    'GAM1': np.exp(0.638641),  # θ9
    'RESD': np.exp(2.08283),  # θ5
    'V1': params_pk['V1'], 'V2': params_pk['V2'], 'V3': params_pk['V3'],
    'CL': params_pk['CL'], 'Q2': params_pk['Q2'], 'Q3': params_pk['Q3'],
}

# Conventional propofol infusion
t_infusion = 10.0 / 60.0
t_interval = 30.0 / 60.0
t_inf_start = 0.0
t_inf_end = - 10.0

# PID parameters
Kp = 2.0
Ki = 0.2
Kd = 0.1

# Parameters for adaptive PID control
Kpmin, Kpmax = 1.0, 5.0
Kimin, Kimax = 0.1, 0.3
Kdmin, Kdmax = 0.00, 0.10

Kp_alpha = 0.0001
Ki_alpha = 0.0
Kd_alpha = 0.0

Kp_log, Ki_log, Kd_log = [Kp], [Ki], [Kd]

BIS_target = 50.0  # target

# Simulation dimension
t_end = 20.0
dt = 0.01
time_points = np.arange(0, t_end, dt)

# IC
q1 = [0.0]
q2 = [0.0]
q3 = [0.0]
u = [0.0]
ce = [0.0]
BIS = [params['EMAX']]

u_values = []
e_integral = 0.0
e_previous = BIS[0] - BIS_target

# 시뮬레이션 루프 (2차 Runge-Kutta)
for i in range(1, len(time_points)):
    t = time_points[i]

    # initialize
    ce_current = ce[-1]
    q1_current = q1[-1]
    q2_current = q2[-1]
    q3_current = q3[-1]

    y = [q1_current, q2_current, q3_current, ce_current]

    BIS_current = pd.calculate_ipred(ce_current, params)
    BIS.append(BIS_current)

    # PID control
    e = BIS_current - BIS_target
    e_integral += e * dt
    e_derivative = (e - e_previous) / dt
    e_previous = e

    u = Kp * e + Ki * e_integral + Kd * e_derivative
    u = max(u, 0.0)
    # u = min(u, 120.0)

    """# conventional method
    if (BIS_current > 50.0 and t < 3.0):
        if (u == 0.0 and t > t_inf_end + t_interval):
            t_inf_start = t
            t_inf_end = t_inf_start + t_infusion
            u = 240.0   # 40mg q10sec
        if (t < t_inf_end):
            u = 240.0
        elif (t >= t_inf_end and t < t_inf_end + t_interval):
            u = 0.0
        else:
            u = 0.0

    elif (BIS_current > 50.0 and t >= 3.0):
        u = 0.2 * wgt
    elif (BIS_current < 50.0 and t >= 3.0):
        u = 0.1 * wgt
    else:
        u = 0.0"""

    u_values.append(u)

    # y1 = pd.pd_model(y, params)
    y1 = pd.pd_model_control(y, u, params)
    k1_q1, k1_q2, k1_q3, k1_ce = y1

    #y2 = pd.pd_model(y1, params)
    y = [q1_current + dt * k1_q1, q2_current + dt * k1_q2, q3_current + dt * k1_q3, ce_current + dt * k1_ce]

    y2 = pd.pd_model_control(y, u, params)
    k2_q1, k2_q2, k2_q3, k2_ce = y2

    q1_next = q1_current + dt / 2.0 * (k1_q1 + k2_q1)
    q2_next = q2_current + dt / 2.0 * (k1_q2 + k2_q2)
    q3_next = q3_current + dt / 2.0 * (k1_q3 + k2_q3)
    ce_next = ce_current + dt / 2.0 * (k1_ce + k2_ce)

    """# APID (todo)
    # increase Kp, Ki for high e (for fast control)
    # increase Kd for high e_derivative
    Kp += Kp_alpha * e * e_derivative
    Ki += Ki_alpha * e * e_integral
    Kd += Kd_alpha * e * e_derivative
    Kp = np.clip(Kp, Kpmin, Kpmax)
    Ki = np.clip(Ki, Kimin, Kimax)
    Kd = np.clip(Kd, Kdmin, Kdmax)"""

    q1.append(q1_next)
    q2.append(q2_next)
    q3.append(q3_next)
    ce.append(ce_next)
    Ki_log.append(Ki)
    Kp_log.append(Kp)
    Kd_log.append(Kd)

# 결과 정리
BIS = np.array(BIS)
u_values = np.array([u_values[0]] + u_values)  # 시간 길이와 맞추기
q1 = np.array(q1)
q2 = np.array(q2)
q3 = np.array(q3)
ce = np.array(ce)

# 그래프 출력
plt.figure(figsize=(12, 5))

# 약물 농도 그래프
plt.subplot(1, 2, 1)
plt.plot(time_points, BIS, label='BIS', color = 'black')
plt.axhline(BIS_target, color='blue', linestyle='--', label='Target BIS')
plt.xlabel('Time (min)')
plt.ylabel('BIS')
plt.title('BIS Over Time')
plt.ylim(0.0, 100.0)
plt.legend()
plt.grid(True)

# 주입 속도 그래프
plt.subplot(1, 2, 2)
plt.plot(time_points, u_values , color='black', label='Infusion Rate u(t)')
plt.xlabel('Time (min)')
plt.ylabel('Infusion Rate (mg/min)')
plt.title('Infusion Rate Over Time')
plt.legend()
plt.ylim(0.0)
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(time_points, q1, color='black', label='primary compartment')
plt.plot(time_points, q2, color='blue', linestyle = '--' ,label='fast compartment')
plt.plot(time_points, q3, color='red', linestyle = '-.' ,label='peripheral compartment')
plt.plot(time_points, ce * params_pk['V1'], color='m', linestyle = ':', linewidth = 3 ,label='effective site')
plt.xlabel('Time (min)')
plt.ylabel('Drug mass (mg)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 5))
plt.plot(time_points, np.array(Kp_log), color='black', label='primary compartment')
plt.plot(time_points, np.array(Ki_log), color='blue', linestyle = '--' ,label='fast compartment')
plt.plot(time_points, np.array(Kd_log), color='red', linestyle = '-.' ,label='peripheral compartment')
plt.xlabel('Time (min)')
plt.ylabel('Drug mass (mg)')
plt.legend()
plt.grid(True)
plt.show()

# save result
output_data = np.column_stack((
    time_points, BIS, u_values,  # u(t) in mg/min
    q1, q2, q3, ce, Kp_log, Ki_log, Kd_log
))

header = 'Time(min)\tBIS\tInfusionRate(mg/min)\tQ1\tQ2\tQ3\tCe\tKp\tKi\tKd\t'
np.savetxt("simulation_output.dat", output_data, delimiter='\t', header=header, comments='', fmt='%.6f')

with open("params.dat", "w") as f:
    f.write("=== general information ===\n")
    f.write(f"age\t{age:6f}\n")
    f.write(f"wgt\t{wgt:6f}\n")
    f.write(f"hgt\t{hgt:6f}\n")
    f.write(f"m1f2\t{m1f2:6f}\n")
    f.write(f"pma\t{pma:6f}\n")
    f.write(f"tech\t{tech:6f}\n")
    f.write(f"a1v2\t{a1v2:6f}\n")
    f.write("=== PD Parameters ===\n")
    for key in sorted(params.keys()):
        f.write(f"{key}\t{params[key]:.6f}\n")
    f.write("=== PK Parameters ===\n")
    for key in sorted(params_pk.keys()):
        f.write(f"{key}\t{params_pk[key]:.6f}\n")
