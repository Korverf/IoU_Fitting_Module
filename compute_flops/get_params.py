import torch
from thop import profile
from modules.iou_fit_conv.iou_fit_conv_bottle_9 import get_model


model = get_model(in_features=16, hidden_features=16)
input1 = torch.randn(1, 8)
input2 = torch.randn(1, 8)
flops, params = profile(model, inputs=(input1, input2))
#print("flops:", flops)
print("params:", params)