from typing import List
import torch
import torch.nn as nn
import copy

max_dist = 2
lam = 0.05
print(f"{max_dist=}, {lam=}")
def recov_distance(x : torch.Tensor, y : torch.Tensor, p=2):
    print("the norm is ", torch.norm(torch.abs(x-y),p=p))
    return torch.norm(torch.abs(x-y),p=p)
def recoverability_loss(layer : nn.Module, x : torch.Tensor, y : torch.Tensor):
    #for loop for every option 4 d skip color dim
    unchange_layer = copy.deepcopy(layer)
    min = 1.1
    bestlist, min_list = [], []
    while min < max_dist:
        min = 2147483647
        best = (-1,-1,-1)
        for i in range(unchange_layer.weight.shape[0]):
            for j in range(unchange_layer.weight.shape[2]):
                for k in range(unchange_layer.weight.shape[3]):
                    if all([(unchange_layer.weight[i,c,j,k].item() == 0) for c in range(unchange_layer.weight.shape[1])]):
                        # print("skipping")
                        continue
                    modif_layer = copy.deepcopy(unchange_layer)
                    modif_layer.weight[i,:,j,k] = 0
                    dist = recov_distance(y, modif_layer(x))
                    if dist < min:
                        best = (i,j,k)
                        min = dist
                        min_layer = modif_layer
                        # print("mid round min", min)
        
        print("final round min",min)
        bestlist.append(best)
        min_list.append((min, min_layer, bestlist.copy()))
        unchange_layer = min_layer
    return min_list

def prune_layer(layer, x, f_max = None, max_score=0):#layer is 4d. blocksize, color , hight, width
    print("before prune:", torch.sum(0==layer.weight))
    min_list = recoverability_loss(layer, x, layer(x))
    f_max = layer if not f_max else f_max
    bl_max = []
    for j, (l,f,b) in enumerate(min_list):
        prune_score = (max_dist-l)/((len(min_list)-j)**(1+lam))
        print(f"{prune_score=}")
        if prune_score > max_score:
            max_score, f_max, bl_max = prune_score, f, b
    # for obj in min_list:
    #     print(obj[2])
    print("the final best prune was: ", bl_max)
    print("final best prune length was: ", len(bl_max))
    return f_max, max_score