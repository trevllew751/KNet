from KNet_RNN import KNet_RNN
from SystemModel import SysModel
from auv_690_config import f, h, Q, R, m, n, m1x_0, m2x_0
from Pipeline import Pipeline
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# fill with auv_log data
# avl_log = torch.zeros(1, 15)
# for i in range(1, 200):
#     avl_log = torch.cat([avl_log, torch.full((1, 15), i)])

### Data Preprocessing ###

avl_log = pd.read_csv("avl_log.csv")
inertial_nav_x_cols = list(filter(lambda x: "[x]" in x, avl_log.columns))
inertial_nav_imu_cols = list(filter(lambda x: "[imu]" in x, avl_log.columns))
inertial_nav_p_cols = list(filter(lambda x: "[P]" in x, avl_log.columns))[-6:]
# _ = [print(x) for x in inertial_nav_p_cols]
avl_log = avl_log[inertial_nav_x_cols + inertial_nav_p_cols +inertial_nav_imu_cols]
avl_log = avl_log.to_numpy()
# _ = [print(x) for x in avl_log.columns]
avl_log = torch.tensor(avl_log).double()
# print(avl_log)
# print(avl_log[0])

#print(avl_log)
#print(avl_log.size(), " is the size of the auv log")
# auv_log = torch.randn(200, 15)

# print(auv_log.size(0))
print(avl_log[0])
# print(auv_log)

steps = 10
lr = 2e-8
wd = 10

auv_690 = SysModel(f, h, Q, R, m, n)
auv_690.InitSequence(m1x_0, m2x_0)

knet = KNet_RNN(m, n)
PipelineKNet = Pipeline(auv_690, knet, avl_log)
PipelineKNet.Initialize()

PipelineKNet.SetTrainingParams(steps, lr, wd)
PipelineKNet.train()
