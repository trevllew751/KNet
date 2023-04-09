from KNet_RNN import KNet_RNN
from auv_690_config import auv_690
from Pipeline import Pipeline
import torch

# fill with auv_log data
auv_log = torch.zeros(1, 15)
for i in range(1, 200):
    auv_log = torch.cat([auv_log, torch.full((1, 15), i)])

# auv_log = torch.randn(200, 15)

# print(auv_log.size(0))
# print(auv_log[0])
# print(auv_log)

steps = 10
lr = 1e-3
wd = 1e-4

knet = KNet_RNN()
PipelineKNet = Pipeline(auv_690, knet, auv_log)
PipelineKNet.Initialize()
PipelineKNet.SetTrainingParams(steps, lr, wd)
PipelineKNet.train()
