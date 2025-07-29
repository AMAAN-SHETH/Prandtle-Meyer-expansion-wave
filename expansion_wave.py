import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
eta = np.linspace(0, 1, 41)
d_eta = eta[1] - eta[0]

# Physical parameters
gamma = 1.4
Nx = 80
Ny = len(eta)
R = 287
h = 40
theta = np.radians(5.352)
x = 0
E = 10
Cy = 0.6

def prandtl_meyer_function(M, gamma=1.4):
    return (np.sqrt((gamma + 1) / (gamma - 1)) * 
            np.arctan(np.sqrt((gamma - 1) * (M ** 2 - 1) / (gamma + 1)))) - \
            np.arctan(np.sqrt(M ** 2 - 1))

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

# Initialize 2D arrays
rho = np.zeros((Nx+1, Ny))
T = np.zeros_like(rho)
P = np.zeros_like(rho)
u = np.zeros_like(rho)
v = np.zeros_like(rho)
M = np.zeros_like(rho)
x_vals = np.zeros(Nx+1)
h_vals = np.zeros(Nx+1)

# Flux arrays
F1 = np.zeros_like(rho)
F2 = np.zeros_like(rho)
F3 = np.zeros_like(rho)
F4 = np.zeros_like(rho)

# Initial conditions
P[0, :] = 101000
T[0, :] = 286.1
rho[0, :] = 1.23
M[0, :] = 2.0
u[0, :] = M[0, :] * np.sqrt(gamma * R * T[0, :])
v[0, :] = 0.0

F1[0, :] = rho[0, :] * u[0, :]
F2[0, :] = rho[0, :] * u[0, :]**2 + P[0, :]
F3[0, :] = rho[0, :] * u[0, :] * v[0, :]
F4[0, :] = (gamma * P[0, :] * u[0, :] / (gamma - 1)) + (rho[0, :] * u[0, :] * (u[0, :]**2 + v[0, :]**2) / 2)

# Main computation loop
for i in range(Nx):
    # Calculate mu and streamline angle
    mu = np.arcsin(1 / M[i, :])
    thetaStreamline = np.arctan(np.abs(v[i, :]) / u[i, :])
    
    # Calculate grid spacing
    dy = h * d_eta
    dx = 0.5 * dy / np.max(np.abs(np.tan(thetaStreamline + mu)))
    x += dx
    x_vals[i+1] = x

    # Update channel height
    if x <= E:
        h = 40
    else:
        h = 40 + ((x - E) * np.tan(theta))
    h_vals[i+1] = h

    # Calculate G terms
    G1 = rho[i, :] * F3[i, :] / F1[i, :]
    G2 = F3[i, :]
    G3 = (rho[i, :] * (F3[i, :] / F1[i, :]) ** 2) + F2[i, :] - (F1[i, :] ** 2 / rho[i, :])
    G4 = (gamma * F3[i, :] * (F2[i, :] - (F1[i, :] ** 2 / rho[i, :])) / 
          (F1[i, :] * (gamma - 1))) + \
         (rho[i, :] * F3[i, :] * ((F1[i, :] / rho[i, :]) ** 2 + (F3[i, :] / F1[i, :]) ** 2) / (2 * F1[i, :]))

    # Calculate S terms (artificial viscosity)
    S1 = np.zeros(Ny)
    S2 = np.zeros(Ny)
    S3 = np.zeros(Ny)
    S4 = np.zeros(Ny)
    
    for j in range(1, Ny-1):
        denom = P[i, j+1] + 2*P[i, j] + P[i, j-1]
        factor = Cy * np.abs(P[i, j+1] - 2*P[i, j] + P[i, j-1]) / denom
        
        S1[j] = (F1[i, j+1] - 2*F1[i, j] + F1[i, j-1]) * factor
        S2[j] = (F2[i, j+1] - 2*F2[i, j] + F2[i, j-1]) * factor
        S3[j] = (F3[i, j+1] - 2*F3[i, j] + F3[i, j-1]) * factor
        S4[j] = (F4[i, j+1] - 2*F4[i, j] + F4[i, j-1]) * factor

    # Calculate F_bar terms (predictor step)
    F1_bar = np.zeros(Ny)
    F2_bar = np.zeros(Ny)
    F3_bar = np.zeros(Ny)
    F4_bar = np.zeros(Ny)
    
    for j in range(1, Ny-1):
        eta_value = eta[j]
        deta_dx = 0 if x < E else (1 - eta_value) * np.tan(theta) / h
        
        F1_bar[j] = F1[i, j] + ((deta_dx * (F1[i, j] - F1[i, j+1]) / d_eta + (G1[j] - G1[j+1]) / (h * d_eta)) * dx + S1[j])
        F2_bar[j] = F2[i, j] + ((deta_dx * (F2[i, j] - F2[i, j+1]) / d_eta + (G2[j] - G2[j+1]) / (h * d_eta)) * dx + S2[j])
        F3_bar[j] = F3[i, j] + ((deta_dx * (F3[i, j] - F3[i, j+1]) / d_eta + (G3[j] - G3[j+1]) / (h * d_eta)) * dx + S3[j])
        F4_bar[j] = F4[i, j] + ((deta_dx * (F4[i, j] - F4[i, j+1]) / d_eta + (G4[j] - G4[j+1]) / (h * d_eta)) * dx + S4[j])

    # Boundary conditions for F_bar
    F1_bar[0] = F1[i, 0]
    F2_bar[0] = F2[i, 0]
    F3_bar[0] = F3[i, 0]
    F4_bar[0] = F4[i, 0]
    
    F1_bar[-1] = F1[i, -1]
    F2_bar[-1] = F2[i, -1]
    F3_bar[-1] = F3[i, -1]
    F4_bar[-1] = F4[i, -1]

    # Calculate intermediate variables
    A = (F3_bar ** 2 / (2 * F1_bar)) - F4_bar
    B = gamma * F1_bar * F2_bar / (gamma - 1)
    C = -(gamma + 1) * F1_bar ** 3 / (2 * (gamma - 1))
    
    rho_bar = (-B + np.sqrt(B**2 - 4*A*C)) / (2 * A)
    u_bar = F1_bar / rho_bar
    P_bar = F2_bar - F1_bar * u_bar

    # Calculate G_bar terms
    G1_bar = rho_bar * F3_bar / F1_bar
    G2_bar = F3_bar
    G3_bar = (rho_bar * (F3_bar / F1_bar) ** 2) + F2_bar - (F1_bar ** 2 / rho_bar)
    G4_bar = (gamma * F3_bar * (F2_bar - (F1_bar ** 2 / rho_bar)) / \
             (F1_bar * (gamma - 1))) + \
             (rho_bar * F3_bar * ((F1_bar / rho_bar) ** 2 + (F3_bar / F1_bar) ** 2) / (2 * F1_bar))

    # Calculate S_bar terms
    S1_bar = np.zeros(Ny)
    S2_bar = np.zeros(Ny)
    S3_bar = np.zeros(Ny)
    S4_bar = np.zeros(Ny)
    
    for j in range(1, Ny-1):
        denom = P_bar[j+1] + 2*P_bar[j] + P_bar[j-1]
        factor = Cy * np.abs(P_bar[j+1] - 2*P_bar[j] + P_bar[j-1]) / denom
        
        S1_bar[j] = (F1_bar[j+1] - 2*F1_bar[j] + F1_bar[j-1]) * factor
        S2_bar[j] = (F2_bar[j+1] - 2*F2_bar[j] + F2_bar[j-1]) * factor
        S3_bar[j] = (F3_bar[j+1] - 2*F3_bar[j] + F3_bar[j-1]) * factor
        S4_bar[j] = (F4_bar[j+1] - 2*F4_bar[j] + F4_bar[j-1]) * factor

    # Update F terms (corrector step)
    for j in range(1, Ny-1):
        eta_value = eta[j]
        deta_dx = 0 if x < E else (1 - eta_value) * np.tan(theta) / h
        
        if j == 0:
            F1[i+1, j] = 0.5 * (F1[i, j] + F1_bar[j] + 
                               ((deta_dx * (F1_bar[j] - F1_bar[j+1]) / d_eta + 
                                (G1_bar[j] - G1_bar[j+1]) / (h * d_eta)) * dx)) + S1_bar[j]
            F2[i+1, j] = 0.5 * (F2[i, j] + F2_bar[j] + 
                               ((deta_dx * (F2_bar[j] - F2_bar[j+1]) / d_eta + 
                                (G2_bar[j] - G2_bar[j+1]) / (h * d_eta)) * dx)) + S2_bar[j]
            F3[i+1, j] = 0.5 * (F3[i, j] + F3_bar[j] + 
                               ((deta_dx * (F3_bar[j] - F3_bar[j+1]) / d_eta + 
                                (G3_bar[j] - G3_bar[j+1]) / (h * d_eta)) * dx)) + S3_bar[j]
            F4[i+1, j] = 0.5 * (F4[i, j] + F4_bar[j] + 
                               ((deta_dx * (F4_bar[j] - F4_bar[j+1]) / d_eta + 
                                (G4_bar[j] - G4_bar[j+1]) / (h * d_eta)) * dx)) + S4_bar[j]
        else:
            F1[i+1, j] = 0.5 * (F1[i, j] + F1_bar[j] + ((deta_dx * (F1_bar[j-1] - F1_bar[j]) / d_eta + (G1_bar[j-1] - G1_bar[j]) / (h * d_eta)) * dx)) + S1_bar[j]
            F2[i+1, j] = 0.5 * (F2[i, j] + F2_bar[j] + ((deta_dx * (F2_bar[j-1] - F2_bar[j]) / d_eta + (G2_bar[j-1] - G2_bar[j]) / (h * d_eta)) * dx)) + S2_bar[j]
            F3[i+1, j] = 0.5 * (F3[i, j] + F3_bar[j] + ((deta_dx * (F3_bar[j-1] - F3_bar[j]) / d_eta + (G3_bar[j-1] - G3_bar[j]) / (h * d_eta)) * dx)) + S3_bar[j]
            F4[i+1, j] = 0.5 * (F4[i, j] + F4_bar[j] + ((deta_dx * (F4_bar[j-1] - F4_bar[j]) / d_eta + (G4_bar[j-1] - G4_bar[j]) / (h * d_eta)) * dx)) + S4_bar[j]

    # Boundary conditions for F
    F1[i+1, 0] = F1[i, 0]
    F2[i+1, 0] = F2[i, 0]
    F3[i+1, 0] = F3[i, 0]
    F4[i+1, 0] = F4[i, 0]
    
    F1[i+1, -1] = F1[i, -1]
    F2[i+1, -1] = F2[i, -1]
    F3[i+1, -1] = F3[i, -1]
    F4[i+1, -1] = F4[i, -1]

    # Update flow variables
    A = (F3[i+1, :] ** 2 / (2 * F1[i+1, :])) - F4[i+1, :]
    B = gamma * F1[i+1, :] * F2[i+1, :] / (gamma - 1)
    C = -(gamma + 1) * F1[i+1, :] ** 3 / (2 * (gamma - 1))
    
    rho[i+1, :] = (-B + np.sqrt(B**2 - 4*A*C)) / (2 * A)
    u[i+1, :] = F1[i+1, :] / rho[i+1, :]
    v[i+1, :] = F3[i+1, :] / F1[i+1, :]
    P[i+1, :] = F2[i+1, :] - (F1[i+1, :] * u[i+1, :])
    T[i+1, :] = P[i+1, :] / (R * rho[i+1, :])
    M[i+1, :] = np.sqrt(u[i+1, :]**2 + v[i+1, :]**2) / np.sqrt(gamma * R * T[i+1, :])

    # Wall boundary condition
    shi = np.arctan(np.abs(v[i+1, 0]) / u[i+1, 0])
    if x < E:
        phi = shi
    else:
        phi = theta - shi

    f_cal = prandtl_meyer_function(M[i+1, 0])
    f_act = f_cal + phi

    M_act = solve_bisection(f_act)
    P_act = P[i+1, 0] * ((1 + ((gamma-1)/2)*M[i+1, 0]**2) / (1 + ((gamma-1)/2)*M_act**2))**(gamma/(gamma-1))
    T_act = T[i+1, 0] * ((1 + ((gamma-1)/2)*M[i+1, 0]**2) / (1 + ((gamma-1)/2)*M_act**2))
    rho_act = P_act / (R * T_act)

    M[i+1, 0] = M_act
    P[i+1, 0] = P_act
    T[i+1, 0] = T_act
    rho[i+1, 0] = rho_act

    if x < E:
        v[i+1, 0] = 0
    else:
        v[i+1, 0] = -u[i+1, 0] * np.tan(theta)

    F1[i+1, 0] = rho[i+1, 0] * u[i+1, 0]
    F2[i+1, 0] = (rho[i+1, 0] * u[i+1, 0]**2) + P[i+1, 0]
    F3[i+1, 0] = rho[i+1, 0] * u[i+1, 0] * v[i+1, 0]
    F4[i+1, 0] = (gamma * P[i+1, 0] * u[i+1, 0] / (gamma-1)) + (rho[i+1, 0] * u[i+1, 0] * (u[i+1, 0]**2 + v[i+1, 0]**2)/2)

# Create physical domain coordinates using the transformation
X_phys = np.zeros((Nx+1, Ny))
Y_phys = np.zeros((Nx+1, Ny))

for i in range(Nx+1):
    x = x_vals[i]
    if x <= E:
        h = 40
        y_a = 0  # Lower wall is flat before expansion
    else:
        h = 40 + (x - E) * np.tan(theta)
        y_a = -(x - E) * np.tan(theta)  # Ramp starts at x = E
    
    for j in range(Ny):
        eta_j = eta[j]
        X_phys[i,j] = x
        Y_phys[i,j] = y_a + eta_j * h  # Transform back to physical coordinates

# Print final values for verification
print("Final x value:", x_vals[-1])
print("\nFinal Mach number array:")
print(M[-1, :])
print("\nFinal u velocity array (m/s):")
print(u[-1, :])
print("\nFinal v velocity array (m/s):")
print(v[-1, :])
print("\nFinal temperature array (K):")
print(T[-1, :])
print("\nFinal density array (kg/m³):")
print(rho[-1, :])

# Plotting in physical domain
plt.figure(figsize=(15, 10))

# Mach Number
plt.subplot(2, 3, 1)
plt.contourf(X_phys, Y_phys, M, levels=20, cmap='jet')
plt.plot(X_phys[:,0], Y_phys[:,0], 'k-', linewidth=2)  # Lower wall
plt.plot(X_phys[:,-1], Y_phys[:,-1], 'k-', linewidth=2)  # Upper wall
plt.colorbar()
plt.title('Mach Number')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

# Pressure
plt.subplot(2, 3, 2)
plt.contourf(X_phys, Y_phys, P, levels=20, cmap='jet')
plt.plot(X_phys[:,0], Y_phys[:,0], 'k-', linewidth=2)
plt.plot(X_phys[:,-1], Y_phys[:,-1], 'k-', linewidth=2)
plt.colorbar()
plt.title('Pressure (Pa)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

# Temperature
plt.subplot(2, 3, 3)
plt.contourf(X_phys, Y_phys, T, levels=20, cmap='jet')
plt.plot(X_phys[:,0], Y_phys[:,0], 'k-', linewidth=2)
plt.plot(X_phys[:,-1], Y_phys[:,-1], 'k-', linewidth=2)
plt.colorbar()
plt.title('Temperature (K)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

# Density
plt.subplot(2, 3, 4)
plt.contourf(X_phys, Y_phys, rho, levels=20, cmap='jet')
plt.plot(X_phys[:,0], Y_phys[:,0], 'k-', linewidth=2)
plt.plot(X_phys[:,-1], Y_phys[:,-1], 'k-', linewidth=2)
plt.colorbar()
plt.title('Density (kg/m³)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

# u Velocity
plt.subplot(2, 3, 5)
plt.contourf(X_phys, Y_phys, u, levels=20, cmap='jet')
plt.plot(X_phys[:,0], Y_phys[:,0], 'k-', linewidth=2)
plt.plot(X_phys[:,-1], Y_phys[:,-1], 'k-', linewidth=2)
plt.colorbar()
plt.title('u Velocity (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

# v Velocity
plt.subplot(2, 3, 6)
plt.contourf(X_phys, Y_phys, v, levels=20, cmap='jet')
plt.plot(X_phys[:,0], Y_phys[:,0], 'k-', linewidth=2)
plt.plot(X_phys[:,-1], Y_phys[:,-1], 'k-', linewidth=2)
plt.colorbar()
plt.title('v Velocity (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.tight_layout()
# plotting the computational domain

plt.figure(figsize=(15, 10))
X, Y = np.meshgrid(x_vals, eta)

plt.subplot(2, 3, 1)
plt.contourf(X, Y, M.T, levels=20, cmap='jet')
plt.colorbar()
plt.title('Mach Number')
plt.xlabel('x (m)')
plt.ylabel('η')

plt.subplot(2, 3, 2)
plt.contourf(X, Y, P.T, levels=20, cmap='jet')
plt.colorbar()
plt.title('Pressure (Pa)')
plt.xlabel('x (m)')
plt.ylabel('η')

plt.subplot(2, 3, 3)
plt.contourf(X, Y, T.T, levels=20, cmap='jet')
plt.colorbar()
plt.title('Temperature (K)')
plt.xlabel('x (m)')
plt.ylabel('η')

plt.subplot(2, 3, 4)
plt.contourf(X, Y, rho.T, levels=20, cmap='jet')
plt.colorbar()
plt.title('Density (kg/m³)')
plt.xlabel('x (m)')
plt.ylabel('η')

plt.subplot(2, 3, 5)
plt.contourf(X, Y, u.T, levels=20, cmap='jet')
plt.colorbar()
plt.title('u Velocity (m/s)')
plt.xlabel('x (m)')
plt.ylabel('η')

plt.subplot(2, 3, 6)
plt.contourf(X, Y, v.T, levels=20, cmap='jet')
plt.colorbar()
plt.title('v Velocity (m/s)')
plt.xlabel('x (m)')
plt.ylabel('η')

plt.tight_layout()
plt.show()