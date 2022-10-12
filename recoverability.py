import torch
import torch.nn as nn
import copy
def recov_distance(x : torch.Tensor, y : torch.Tensor, p=2):
    return torch.norm(torch.abs(x-y),p=p)
def recoverability_loss(layer : nn.Module, x : torch.Tensor, y : torch.Tensor):
    #for loop for every option
    modif_layer = copy.deepcopy(layer)
    min = 2147483647
    #using as two dementional check that it really is 
    #how to go over all option
    dist = recov_distance(y,modif_layer(x))
    if dist < min:
        min = dist
        min_latyer = modif_layer
    return min, min_latyer