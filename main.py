from KNet_RNN import KNet_RNN
from auv_690_config import auv_690
from Pipeline import Pipeline
import pandas as pd
import numpy as np
import torch

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

print(auv_log)
print(auv_log.size(), " is the size of the auv log")
# auv_log = torch.randn(200, 15)

# print(auv_log.size(0))
# print(auv_log[0])
# print(auv_log)

steps = 10
lr = 1e-3
wd = 1e-4

knet = KNet_RNN()
PipelineKNet = Pipeline(auv_690, knet, avl_log)
PipelineKNet.Initialize()
PipelineKNet.SetTrainingParams(steps, lr, wd)
PipelineKNet.train()
