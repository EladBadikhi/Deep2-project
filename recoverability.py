from typing import List
import torch
import torch.nn as nn
import copy

max_dist = 2
lam = 0.6
span = 32
print(f"{max_dist=}, {lam=}, {span=}")
def recov_distance(x : torch.Tensor, y : torch.Tensor, p=2):
    return torch.norm(torch.abs(x-y),p=p,dim=(1,2,3)).max()
def recoverability_loss(layer : nn.Module, x : torch.Tensor, y : torch.Tensor):
    #for loop for every option 4 d skip color dim
    modif_layer = copy.deepcopy(layer)
    min_dist = 1.1
    bestlist, min_list = [], []
    import timeit
    staime = timeit.default_timer()
    while min_dist < max_dist:
        roundstaime = timeit.default_timer()
        min_dist = 2147483647
        best = (-1,-1,-1,-1)
        min = span if modif_layer.weight.shape[0] > span else modif_layer.weight.shape[0]
        for i in range(0,modif_layer.weight.shape[0],min+1):
            min = span if modif_layer.weight.shape[0] - i > span else modif_layer.weight.shape[0] - i
            for j in range(modif_layer.weight.shape[2]):
                for k in range(modif_layer.weight.shape[3]):
                    for c in range(0,modif_layer.weight.shape[1]):
                        if all([(modif_layer.weight[iter,c,j,k].item() == 0) for iter in range(i,i+min)]):#torch.all((modif_layer.weight[i,c:c+min,j,k] == 0)).item():
                            # print("skipping ",min)
                            continue
                        modif_layer.weight[i:i+min,c,j,k] = 0
                        dist = recov_distance(y, modif_layer(x))
                        if dist < min_dist:
                            best = (i,c,min,j,k)
                            min_dist = dist
                        modif_layer.weight[i:i+min,c,j,k] = layer.weight[i:i+min,c,j,k] 
        
        print(f"time for round {timeit.default_timer()-roundstaime}" )
        print("final round min",min_dist, best)
        bestlist.append(best)
        min_list.append((min_dist, bestlist.copy()))
        fi, fc, fcp, fj, fk = best
        modif_layer.weight[fi:fi+fcp,fc,fj,fk] = 0
        #....................................................................
        # for i in range(0,modif_layer.weight.shape[0]):
        #     for j in range(modif_layer.weight.shape[2]):
        #         for k in range(modif_layer.weight.shape[3]):
        #             min = span if modif_layer.weight.shape[1] > span else modif_layer.weight.shape[1]
        #             for c in range(0,modif_layer.weight.shape[1],min+1):
        #                 min = span if modif_layer.weight.shape[1] - c > span else modif_layer.weight.shape[1] - c
        #                 if all([(modif_layer.weight[i,iter,j,k].item() == 0) for iter in range(c,c+min)]):#torch.all((modif_layer.weight[i,c:c+min,j,k] == 0)).item():
        #                     # print("skipping ",min)
        #                     continue
        #                 modif_layer.weight[i,c:c+min,j,k] = 0
        #                 dist = recov_distance(y, modif_layer(x))
        #                 if dist < min_dist:
        #                     best = (i,c,min,j,k)
        #                     min_dist = dist
        #                 modif_layer.weight[i,c:c+min,j,k] = layer.weight[i,c:c+min,j,k]
        
        # print(f"time for round {timeit.default_timer()-roundstaime}" )
        # print("final round min",min_dist, best)
        # bestlist.append(best)
        # min_list.append((min_dist, bestlist.copy()))
        # fi, fc, fcp, fj, fk = best
        # modif_layer.weight[fi,fc:fc+fcp,fj,fk] = 0
    print(f"time for all round for layer {timeit.default_timer()-staime}" )
    return min_list

def prune_layer(layer, x, f_max = None, max_score=0):#layer is 4d. blocksize, color , hight, width
    print("before prune:", torch.sum(0==layer.weight))
    min_list = recoverability_loss(layer, x, layer(x))
    f_max = layer if not f_max else f_max
    bl_max = []
    for j, (l,b) in enumerate(min_list):
        prune_score = (max_dist-l)/((len(min_list)-j)**(1+lam))
        # print(f"{prune_score=}")
        if prune_score > max_score:
            max_score, bl_max = prune_score, b
    for fi, fc, fcp, fj, fk in b:
        f_max.weight[fi,fc:fc+fcp,fj,fk] = 0
    print("the final best prune was: ", bl_max)
    print("final best prune length was: ", len(bl_max))
    return f_max, max_score