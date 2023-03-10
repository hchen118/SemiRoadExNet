import torch
import warnings

warnings.filterwarnings("ignore")
smooth = 1e-8

def cal_entropy(gx):
    ex = -1.0 * gx*torch.log(gx + smooth) - 1.0 * (1-gx)*torch.log(1-gx+smooth)
    return ex


