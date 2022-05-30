import torch
from modules.iou_fit_conv.iou_fit_conv_bottle_9 import get_model
from pthflops import count_ops

# Create a network and a corresponding input
device = 'cuda:0'
model = get_model(in_features=16, hidden_features=16).to(device)
inp1 = torch.rand(1,8).to(device)
inp2 = torch.rand(1,8).to(device)
# Count the number of FLOPs
count_ops(model, (inp1,inp2))