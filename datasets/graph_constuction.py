import torch


def create_graph(coordinates, mask, k):
    ori_cor = coordinates.view(-1, 2)

    coorX = coordinates[:, :, 0].detach().clone()
    coorY = coordinates[:, :, 1].detach().clone()

    Observe_coorX = coorX.masked_fill(mask, value=0)
    Observe_coorY = coorY.masked_fill(mask, value=0)

    # get target nodes coordinates
    Observe_coorX = Observe_coorX[Observe_coorX != 0].view(-1, 1)
    Observe_coorY = Observe_coorY[Observe_coorY != 0].view(-1, 1)

    Observe_coor = torch.cat((Observe_coorX, Observe_coorY), dim=-1)

    Target_coorX = coorX.masked_fill(~mask, value=0)
    Target_coorY = coorY.masked_fill(~mask, value=0)

    # get target nodes coordinates
    Target_coorX = Target_coorX[Target_coorX != 0].view(-1, 1)
    Target_coorY = Target_coorY[Target_coorY != 0].view(-1, 1)

    Target_coor = torch.cat((Target_coorX, Target_coorY), dim=-1)

    distances = torch.sqrt(((Target_coor[:, None] - Observe_coor) ** 2).sum(-1))
    nearst_points = distances.topk(k, dim=1, largest=False)[1]
    edge_index = create_edge(nearst_points, Observe_coor, Target_coor, ori_cor)

    distances_obs = torch.sqrt(((Observe_coor[:, None] - Observe_coor) ** 2).sum(-1))
    nearst_points_obs = distances_obs.topk(k + 1, dim=1, largest=False)[1][:, 1:]
    edge_index_obs = create_edge(nearst_points_obs, Observe_coor, Observe_coor, ori_cor)

    distances_tar = torch.sqrt(((Target_coor[:, None] - Target_coor) ** 2).sum(-1))
    nearst_points_tar = distances_tar.topk(k + 1, dim=1, largest=False)[1][:, 1:]
    edge_index_tar = create_edge(nearst_points_tar, Target_coor, Target_coor, ori_cor)
    # print(coorX)

    return torch.cat((edge_index, edge_index_obs, edge_index_tar), dim=-1)


def create_edge(neighbors, Observe_coor, Target_coor, ori_cor):
    row = []  # source
    col = []  # target

    for cor, neis in zip(Target_coor, neighbors):
        tar_idx = (ori_cor == cor).all(dim=1).nonzero().squeeze().item()
        for n in neis:
            nei_idx = (ori_cor == Observe_coor[n]).all(dim=1).nonzero().squeeze().item()
            if Observe_coor[n][0] < cor[0]:
                row.append(nei_idx)
                col.append(tar_idx)
            else:
                row.append(tar_idx)
                col.append(nei_idx)

    return torch.tensor([row, col])
