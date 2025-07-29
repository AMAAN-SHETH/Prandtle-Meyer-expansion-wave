import numpy as np
import matplotlib.pyplot as plt

eta = np.linspace(0, 1, 41)
y = np.zeros_like(eta)
d_eta = eta[1] - eta[0]

gamma = 1.4
Nx = 80 # iterations
R = 287
h = 40  # initial height in y
theta = np.radians(5.352) # exapansion corner
x = 0
E = 10 # distance of expansion corner
Cy = 0.6 # courant number


def prandtl_meyer_function(M, gamma=1.4):
    return (np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) * (M ** 2 - 1) / (gamma + 1)))) - np.arctan(np.sqrt(M ** 2 - 1))


def solve_bisection(f_cal, gamma=1.4, tol=1e-6):
    M_low = 1.01
    M_high = 10

    f_low = prandtl_meyer_function(M_low, gamma) - f_cal
    f_high = prandtl_meyer_function(M_high, gamma) - f_cal
    if f_low * f_high > 0:
        raise ValueError("Root is out of bounds")

    while (M_high - M_low) > tol:
        M_mid = 0.5 * (M_low + M_high)
        f_mid = prandtl_meyer_function(M_mid, gamma) - f_cal

        if f_mid == 0 or (M_high - M_low) < tol:
            return M_mid
        elif f_mid * f_low < 0:
            M_high = M_mid
        else:
            M_low = M_mid

    return 0.5 * (M_low + M_high)


rho = np.zeros_like(eta)
T = np.zeros_like(eta)
P = np.zeros_like(eta)
u = np.zeros_like(eta)
v = np.zeros_like(eta)
M = np.zeros_like(eta)

S1 = np.zeros_like(eta)
S2 = np.zeros_like(eta)
S3 = np.zeros_like(eta)
S4 = np.zeros_like(eta)

S1_bar = np.copy(S1)
S2_bar = np.copy(S2)
S3_bar = np.copy(S3)
S4_bar = np.copy(S4)

# initial condition
P.fill(101000)
T.fill(286.1)
rho.fill(1.23)
M.fill(2)
u = M * np.sqrt(gamma * R * T)

# fluxes
F1 = rho * u
F2 = (rho * u ** 2) + P
F3 = rho * u * v
F4 = (gamma * P * u / (gamma - 1)) + (rho * u * (u ** 2 + v ** 2) / 2)

for i in range(Nx):

    # spatial step using CFL
    mu = np.arcsin(1 / M)
    dy = h * d_eta
    dx = 0.5 * dy / np.abs(np.max(np.tan(theta + mu)))
    x = x + dx
    # print(dx)

    # adjusted h according to x
    if x <= 10:
        h = 40
    else:
        h = 40 + ((x - 10) * np.tan(theta))
    

    G1 = rho * F3 / F1
    G2 = F3
    G3 = (rho * (F3 / F1) ** 2) + F2 - (F1 ** 2 / rho)
    G4 = (gamma * F3 * (F2 - (F1 ** 2 / rho)) / (F1 * (gamma - 1))) + (rho * F3 * ((F1 / rho) ** 2 + (F3 / F1) ** 2) / (2 * F1))

    F1_bar = np.copy(F1)
    F2_bar = np.copy(F2)
    F3_bar = np.copy(F3)
    F4_bar = np.copy(F4)

    # artificial viscocity
    for j in range(len(eta) - 1):
        S1[j] = Cy * (F1[j + 1] - 2 * F1[j] + F1[j - 1]) * np.abs(P[j + 1] - 2 * P[j] + P[j - 1]) / (P[j + 1] + 2 * P[j] + P[j - 1])
        S2[j] = Cy * (F2[j + 1] - 2 * F2[j] + F2[j - 1]) * np.abs(P[j + 1] - 2 * P[j] + P[j - 1]) / (P[j + 1] + 2 * P[j] + P[j - 1])
        S3[j] = Cy * (F3[j + 1] - 2 * F3[j] + F3[j - 1]) * np.abs(P[j + 1] - 2 * P[j] + P[j - 1]) / (P[j + 1] + 2 * P[j] + P[j - 1])
        S4[j] = Cy * (F4[j + 1] - 2 * F4[j] + F4[j - 1]) * np.abs(P[j + 1] - 2 * P[j] + P[j - 1]) / (P[j + 1] + 2 * P[j] + P[j - 1])

    # predictor step
    for j in range(len(eta) - 1):
        eta_value = eta[j + 1]
        # metrics calculations
        dxidx = 1
        if x < E:
            deta_dx = 0
        else:
            deta_dx = (1 - eta_value) * np.tan(theta) / h
        deta_dy = 1 / h
        # predictor
        F1_bar[j] = F1[j] + (((deta_dx * (F1[j] - F1[j + 1]) / d_eta) + ((G1[j] - G1[j + 1]) / (h * d_eta))) * dx) + S1[j]
        F2_bar[j] = F2[j] + (((deta_dx * (F2[j] - F2[j + 1]) / d_eta) + ((G2[j] - G2[j + 1]) / (h * d_eta))) * dx) + S2[j]
        F3_bar[j] = F3[j] + (((deta_dx * (F3[j] - F3[j + 1]) / d_eta) + ((G3[j] - G3[j + 1]) / (h * d_eta))) * dx) + S3[j]
        F4_bar[j] = F4[j] + (((deta_dx * (F4[j] - F4[j + 1]) / d_eta) + ((G4[j] - G4[j + 1]) / (h * d_eta))) * dx) + S4[j]

    A = (F3_bar ** 2 / (2 * F1_bar)) - F4_bar
    B = gamma * F1_bar * F2_bar / (gamma - 1)
    C = -(gamma + 1) * F1_bar ** 3 / (2 * (gamma - 1))
    # deconstruction of primitives
    rho_bar = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    G1_bar = rho_bar * F3_bar / F1_bar
    G2_bar = F3_bar
    G3_bar = (rho_bar * (F3_bar / F1_bar) ** 2) + F2_bar - (F1_bar ** 2 / rho_bar)
    G4_bar = (gamma * F3_bar * (F2_bar - (F1_bar ** 2 / rho_bar)) / (F1_bar * (gamma - 1))) + (rho_bar * F3_bar * ((F1_bar / rho_bar) ** 2 + (F3_bar / F1_bar) ** 2) / (2 * F1_bar))

    u_bar = F1_bar / rho_bar
    P_bar = F2_bar - (F1_bar * u_bar)
    # artificial viscocity in bar
    for j in range(len(eta) - 1):
        S1_bar[j] = Cy * (F1_bar[j + 1] - 2 * F1_bar[j] + F1_bar[j - 1]) * np.abs(P_bar[j + 1] - 2 * P_bar[j] + P_bar[j - 1]) / (P_bar[j + 1] + 2 * P_bar[j] + P_bar[j - 1])
        S2_bar[j] = Cy * (F2_bar[j + 1] - 2 * F2_bar[j] + F2_bar[j - 1]) * np.abs(P_bar[j + 1] - 2 * P_bar[j] + P_bar[j - 1]) / (P_bar[j + 1] + 2 * P_bar[j] + P_bar[j - 1])
        S3_bar[j] = Cy * (F3_bar[j + 1] - 2 * F3_bar[j] + F3_bar[j - 1]) * np.abs(P_bar[j + 1] - 2 * P_bar[j] + P_bar[j - 1]) / (P_bar[j + 1] + 2 * P_bar[j] + P_bar[j - 1])
        S4_bar[j] = Cy * (F4_bar[j + 1] - 2 * F4_bar[j] + F4_bar[j - 1]) * np.abs(P_bar[j + 1] - 2 * P_bar[j] + P_bar[j - 1]) / (P_bar[j + 1] + 2 * P_bar[j] + P_bar[j - 1])
    # corrector(forward for boundary, and backward for other points)
    for j in range(len(eta) - 1):
        eta_value = eta[j + 1]

        dxidx = 1
        if x < E:
            deta_dx = 0
        else:
            deta_dx = (1 - eta_value) * np.tan(theta) / h
        deta_dy = 1 / h
        # boundary points
        if j == 0:
            F1[j] = 0.5 * (F1[j] + F1_bar[j] + (((deta_dx * (F1_bar[j] - F1_bar[j + 1]) / d_eta) + ((G1_bar[j] - G1_bar[j + 1]) / (h * d_eta))) * dx)) + S1_bar[j]
            F2[j] = 0.5 * (F2[j] + F2_bar[j] + (((deta_dx * (F2_bar[j] - F2_bar[j + 1]) / d_eta) + ((G2_bar[j] - G2_bar[j + 1]) / (h * d_eta))) * dx)) + S2_bar[j]
            F3[j] = 0.5 * (F3[j] + F3_bar[j] + (((deta_dx * (F3_bar[j] - F3_bar[j + 1]) / d_eta) + ((G3_bar[j] - G3_bar[j + 1]) / (h * d_eta))) * dx)) + S3_bar[j]
            F4[j] = 0.5 * (F4[j] + F4_bar[j] + (((deta_dx * (F4_bar[j] - F4_bar[j + 1]) / d_eta) + ((G4_bar[j] - G4_bar[j + 1]) / (h * d_eta))) * dx)) + S4_bar[j]
        # rest of the points
        else:
            F1[j] = 0.5 * (F1[j] + F1_bar[j] + (((deta_dx * (F1_bar[j - 1] - F1_bar[j]) / d_eta) + ((G1_bar[j - 1] - G1_bar[j]) / (h * d_eta))) * dx)) + S1_bar[j]
            F2[j] = 0.5 * (F2[j] + F2_bar[j] + (((deta_dx * (F2_bar[j - 1] - F2_bar[j]) / d_eta) + ((G2_bar[j - 1] - G2_bar[j]) / (h * d_eta))) * dx)) + S2_bar[j]
            F3[j] = 0.5 * (F3[j] + F3_bar[j] + (((deta_dx * (F3_bar[j - 1] - F3_bar[j]) / d_eta) + ((G3_bar[j - 1] - G3_bar[j]) / (h * d_eta))) * dx)) + S3_bar[j]
            F4[j] = 0.5 * (F4[j] + F4_bar[j] + (((deta_dx * (F4_bar[j - 1] - F4_bar[j]) / d_eta) + ((G4_bar[j - 1] - G4_bar[j]) / (h * d_eta))) * dx)) + S4_bar[j]

    A = (F3 ** 2 / (2 * F1)) - F4
    B = gamma * F1 * F2 / (gamma - 1)
    C = -(gamma + 1) * F1 ** 3 / (2 * (gamma - 1))
    # premitives
    rho = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    u = F1 / rho
    v = F3 / F1
    P = F2 - (F1 * u)
    T = P / (R * rho)
    M = np.sqrt(u ** 2 + v ** 2) / np.sqrt(gamma * R * T)
    print(rho)
    # abberts method on boundary
    shi = np.arctan(np.abs(v[0]) / u[0])
    phi = theta - shi
    f_cal = np.radians(np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) * (M[0] ** 2 - 1) / (gamma + 1))) - np.arctan(np.sqrt(M[0] ** 2 - 1)))
    f_act = f_cal + phi

    M_act = solve_bisection(f_act)
    P_act = P[0] * ((1 + (((gamma - 1) / 2) * M[0] ** 2)) / (1 + (((gamma - 1) / 2) * M_act ** 2))) ** (gamma / (gamma - 1))
    T_act = T[0] * ((1 + (((gamma - 1) / 2) * M[0] ** 2)) / (1 + (((gamma - 1) / 2) * M_act ** 2)))
    rho_act = P_act / (R * T_act)
    # updating the boundaries
    M[0] = M_act
    P[0] = P_act
    T[0] = T_act
    rho[0] = rho_act
    v[0] = -u[0] * np.tan(theta)
    # updating the fluxes
    F1 = rho * u
    F2 = (rho * u ** 2) + P
    F3 = rho * u * v
    F4 = (gamma * P * u / (gamma - 1)) + (rho * u * (u ** 2 + v ** 2) / 2)

# final results
print(u)
print(v)
print(rho)
print(P)
print(T)
print(M)
print(eta)
print(x)
