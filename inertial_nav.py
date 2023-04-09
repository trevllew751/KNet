import numpy as np
from avl_matrix import skew, orthonormalize

omega_ie = 7.292115E-5
e = 0.0818191908425
R_0 = 6378137.0
R_P = 6356752.31425
mu = 3.986004418E14
f = 1.0 / 298.257223563


def earth_rotation_rate(p_b):
    L_b = p_b[0]
    omega_ie_n = np.array([omega_ie * np.cos(L_b), 0.0, -omega_ie * np.sin(L_b)])
    return omega_ie_n


def transport_rate(p_b, v_eb_n):
    L_b = p_b[0]
    h_b = p_b[2]
    # R_N = 0
    # R_E = 0
    # r_eS_e = 0
    # R_N, R_E, r_eS_e = radii_of_curvature(p_b, R_N, R_E, r_eS_e)
    R_N, R_E, r_eS_e = radii_of_curvature(p_b)

    omega_en_n = np.array([
        v_eb_n[1] / (R_E + h_b),
        -v_eb_n[0] / (R_N + h_b),
        -v_eb_n[1] * np.tan(L_b) / (R_E + h_b)
    ])

    return omega_en_n


def somigliana_gravity(L_b):
    sinsq_L = np.sin(L_b) * np.sin(L_b)
    return 9.7803253359 * (1.0 + 0.001931853 * sinsq_L) / np.sqrt(1.0 - e * e * sinsq_L)


def gravitational_acceleration(p_b):
    L_b = p_b[0]
    h_b = p_b[2]

    g_b_n = np.zeros((3, 1))

    g_0 = somigliana_gravity(L_b)

    g_b_n[0] = -8.08E-9 * h_b * np.sin(2 * L_b)

    g_b_n[1] = 0

    g_b_n[2] = g_0 * (1.0 - (2.0 / R_0) * (1.0 + f * (1.0 - 2.0 * np.sin(L_b) * np.sin(L_b)) +
                                           (omega_ie * omega_ie * R_0 * R_0 * R_P / mu)) * h_b +
                      (3.0 * h_b * h_b / (R_0 * R_0)))

    return g_b_n.reshape((3,))


def radii_of_curvature(p_b):
    L_b = p_b[0]

    sinsq_L = np.sin(L_b) * np.sin(L_b)
    denom = 1.0 - e * e * sinsq_L

    R_N = R_0 * (1.0 - e * e) / pow(denom, 1.5)

    R_E = R_0 / np.sqrt(denom)

    r_eS_e = R_E * np.sqrt(np.cos(L_b) * np.cos(L_b) + (1 - e * e) * (1 - e * e) * np.sin(L_b) * np.sin(L_b))

    return R_N, R_E, r_eS_e


def f_ins_hp(C_b_n, v_eb_n, p_b, w_ib_b, f_ib_b, dt):
    L_b = p_b[0]
    lambda_b = p_b[1]
    h_b = p_b[2]

    I = np.identity(3)

    a_ib_b = w_ib_b * dt
    mag_a = np.linalg.norm(a_ib_b)
    A_ib_b = skew(a_ib_b)
    W_ib_b = skew(w_ib_b)

    w_ie_n = earth_rotation_rate(p_b)
    W_ie_n = skew(w_ie_n)

    w_en_n = transport_rate(p_b, v_eb_n)
    W_en_n = skew(w_en_n)

    g_b_n = gravitational_acceleration(p_b)

    C_bb_bm = np.zeros((3, 3))

    if mag_a > 1.0e-8:
        term1 = (1 - np.cos(mag_a)) / (mag_a * mag_a) * A_ib_b  # 3x3 matrix
        term2 = (1.0 / (mag_a * mag_a)) * (1.0 - np.sin(mag_a) / mag_a) * np.matmul(A_ib_b, A_ib_b)  # 3x3 matrix
        C_bb_bm = I + term1 + term2
    else:
        C_bb_bm = I + W_ib_b * dt

    Cbar_b_n = np.matmul(C_b_n, C_bb_bm) - 0.5 * np.matmul((W_ie_n + W_en_n), C_b_n) * dt

    f_ib_n = np.matmul(Cbar_b_n, f_ib_b)

    v_eb_n_old = v_eb_n

    v_eb_n = v_eb_n + (f_ib_n + g_b_n - np.matmul((W_en_n + 2.0 * W_ie_n), v_eb_n)) * dt    # v_eb_n[2] going to inf

    # print(f"v_eb_n {v_eb_n}")

    # R_N = 0
    # R_E = 0
    # r_eS_e = 0
    # R_N, R_E, r_eS_e = radii_of_curvature(p_b, R_N, R_E, r_eS_e)
    R_N, R_E, r_eS_e = radii_of_curvature(p_b)

    L_b_old = L_b
    h_b_old = h_b

    h_b = h_b - 0.5 * dt * (v_eb_n_old[2] + v_eb_n[2])

    # print(f"h_b {h_b}")

    p_b[2] = h_b    # going to -inf

    L_b = L_b + 0.5 * dt * (v_eb_n_old[0] / (R_N + h_b) + v_eb_n[0] / (R_N + h_b))
    p_b[0] = L_b

    R_E_old = R_E

    R_N, R_E, r_eS_e = radii_of_curvature(p_b)

    term_old = v_eb_n_old[1] / ((R_E_old + h_b_old) * np.cos(L_b_old))
    term_new = v_eb_n[1] / ((R_E + h_b) * np.cos(L_b))
    lambda_b = lambda_b + dt / 2.0 * (term_old + term_new)
    p_b[1] = lambda_b

    W_en_n_old = W_en_n

    w_en_n = transport_rate(p_b, v_eb_n)
    W_en_n = skew(w_en_n)

    C_bp_bm = I + A_ib_b

    C_b_n = np.matmul(np.matmul((I - (W_ie_n + 0.5 * W_en_n_old + 0.5 * W_en_n) * dt), C_b_n), C_bp_bm)

    C_b_n = orthonormalize(C_b_n)

    return C_b_n, v_eb_n, p_b
