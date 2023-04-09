from avl_matrix import *
from inertial_nav import *
import numpy as np


def transformation_function(x, v, w_ib_b, f_ib_b, dt):
    # x = x.astype(np.float32)
    # TODO: temporary
    x = x.detach().numpy()
    v = v.numpy()[0]
    w_ib_b = w_ib_b.numpy()[0]
    f_ib_b = f_ib_b.numpy()[0]

    # print(x)
    # print(v)
    # print(w_ib_b)
    # print(f_ib_b)

    w_g = v[0:3]
    w_a = v[3:]

    C_n_b = euler_to_matrix(x[0:3])  # roll[rad], pitch[rad], yaw[rad]
    v_eb_n = x[3:6]  # v_eb_b_x, v_eb_b_y, v_eb_b_z
    p_b = x[6:9]  # lat[rad], lon[rad], alt [m]
    b_g = x[9:12]  # bgx, bgy, bgz
    b_a = x[12:15]  # bax, bay, baz

    w_ib_b = np.add(w_ib_b, (w_g - b_g))
    f_ib_b = np.add(f_ib_b, (w_a - b_a))

    C_b_n = C_n_b.transpose()

    C_b_n, v_eb_n, p_b = f_ins_hp(C_b_n, v_eb_n, p_b, w_ib_b, f_ib_b, dt)

    C_n_b = C_b_n.transpose()

    x[0:3] = matrix_to_euler(C_n_b)
    x[3:6] = v_eb_n
    x[6:9] = p_b
    x[9:12] = b_g
    x[12:] = b_a

    return x


def test_transformation_function():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    v = np.array([1, 2, 3, 4, 5, 6])
    w_ib_b = np.array([11, 12, 13])
    f_ib_b = np.array([14, 15, 16])
    dt = 6.7
    output = transformation_function(x, v, w_ib_b, f_ib_b, dt)

    print(output)

    x = np.array([62,18,39,72,52,75,43,38,27,77,5,21,6,67,12])
    v = np.array([91,42,29,8,56,16])
    w_ib_b = np.array([58,67,69])
    f_ib_b = np.array([92,83,44])
    dt = 17.0

    output = transformation_function(x, v, w_ib_b, f_ib_b, dt)

    print(output)


def test():
    print(skew(np.array([1, 2, 3])))

# test_transformation_function()
# test()
