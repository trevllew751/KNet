import numpy as np
from avl_math import *


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def euler_to_matrix(theta, degrees=False):
    if (degrees):
        theta[0] = deg_to_rad(theta[0])
        theta[1] = deg_to_rad(theta[1])
        theta[2] = deg_to_rad(theta[2])

    sin_phi = np.sin(theta[0])
    cos_phi = np.cos(theta[0])
    sin_theta = np.sin(theta[1])
    cos_theta = np.cos(theta[1])
    sin_psi = np.sin(theta[2])
    cos_psi = np.cos(theta[2])

    C = np.empty((3, 3))

    C[0, 0] = cos_theta * cos_psi
    C[0, 1] = cos_theta * sin_psi
    C[0, 2] = -sin_theta
    C[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    C[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    C[1, 2] = sin_phi * cos_theta
    C[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    C[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    C[2, 2] = cos_phi * cos_theta

    return C


def matrix_to_euler(R, degrees=False):
    roll = np.arctan2(R[1, 2], R[2, 2])
    pitch = np.arcsin(-R[0, 2])
    yaw = np.arctan2(R[0, 1], R[0, 0])

    if (degrees):
        roll = rad_to_deg(roll)
        pitch = rad_to_deg(pitch)
        yaw = rad_to_deg(yaw)

    theta = np.array([roll, pitch, yaw])
    return theta


def orthonormalize(C):
    c1 = C[0, :]
    c2 = C[1, :]
    c3 = C[2, :]

    D12 = np.matmul(c1.transpose(), c2)
    D23 = np.matmul(c2.transpose(), c3)
    D13 = np.matmul(c1.transpose(), c3)

    c1p = c1 - 0.5 * D12 * c2 - 0.5 * D13 * c3
    c2p = c2 - 0.5 * D12 * c1 - 0.5 * D23 * c3
    c3p = c3 - 0.5 * D13 * c1 - 0.5 * D23 * c2

    c1p = 2.0 / (1.0 + np.matmul(c1p.transpose(), c1p)) * c1p
    c2p = 2.0 / (1.0 + np.matmul(c2p.transpose(), c2p)) * c2p
    c3p = 2.0 / (1.0 + np.matmul(c3p.transpose(), c3p)) * c3p


    C_orthonormalized = np.zeros(C.shape)
    C_orthonormalized[0, :] = c1p
    C_orthonormalized[1, :] = c2p
    C_orthonormalized[2, :] = c3p

    return C_orthonormalized
