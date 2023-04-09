from SystemModel import SysModel
from transformation_function import transformation_function
from observation_function import observation_function
import torch


def fake_func(x, v, w, f, dt):
    ones = torch.ones(1, 15)
    # print(x+ones)
    return x + ones


# f = fake_func
f = transformation_function
h = observation_function
Q = torch.tensor([3.0000e-04, 3.0000e-04, 3.0000e-03,
                  1.6000e-05, 1.6000e-05, 1.6000e-05,
                  1.5398e-13, 1.5398e-13, 1.0000e+00,
                  5.8761e-16, 5.8761e-16, 5.8761e-16,
                  2.1638e-08, 2.1638e-08, 2.1638e-08])
R = torch.zeros(Q.size())

auv_690 = SysModel(f, h, Q, R)
