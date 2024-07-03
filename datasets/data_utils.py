"""
preprocess the datapoints
@Author: Ned
"""

import pickle
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from datasets.scaling import *
from datasets.graph_constuction import create_graph
from datasets.masking_strategies import pixel_level, patch_level

torch.manual_seed(42)


class Dataset(torch.utils.data.Dataset):
    r"""A base class for representing a Dataset.

    Args:
        path (string): Path to the h5 file.
        transform (callable, optional): A function/transform that takes in a :obj:`graphs4cfd.graph.Graph` object
            and returns a transformed version. The data object will be transformed before every access.
            (default: :obj:`None`)
        training_info (dict, optional): A dictionary containing values of type :obj:`ìnt` for the keys `n_in`,
            `n_out`, `step` and `T`. (default: :obj:`None`)
        idx (int, optional): The index of the simulation to load. If :obj:`None`, then all the simulations are loaded.
            (default: :obj:`None`)
        preload (bool, optional): If :obj:`True`, then the data is loaded in memory. If :obj:`False`, then the data
            is loaded from the h5 file at every access. (default: :obj:`False`)
    """

    def __init__(self, path: str, graph: str, scaler: str, normal: bool, K: int, mask_way: str, mask_ratio: float):
        self.path = path
        self.graph = graph
        self.scaler = scaler
        self.normal = normal
        self.K = K
        self.mask_way = mask_way
        self.mask_ratio = mask_ratio

        self.load()

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.datas)

    def __getitem__(self,
                    idx: int):
        r"""Get the idx-th training sample."""
        sample = self.datas[idx]  # [H, W, 5] v_x, v_y, wd, coordinates

        data = sample['x']  # v_x, v_y, label, mask, coordinates
        ext = sample['ext']  # [water_depth, month, weekday, hour]
        ext = ext.reshape(1, -1)

        coordinates = data[:, :, -2:]  # [H, W, 2]

        v_x = data[:, :, 0]
        v_y = data[:, :, 1]

        v_x_ori = v_x.detach().clone()
        v_y_ori = v_y.detach().clone()

        label = data[:, :, 2]  # 0-nan, 1-value -> 0, 1, 2: mask

        if self.scaler != None:
            if self.scaler == 'minmax':
                v_x = minmax(v_x)
                v_y = minmax(v_y)
                # wd = minmaxNorWD(wd)
            elif self.scaler == 'z':
                v_x = zscore(v_x)
                v_y = zscore(v_y)
                # wd = zNorWD(wd)

        # masking
        if 'pixel' in self.mask_way:
            mask = pixel_level(v_x, label, self.mask_ratio)
        else:
            mask = patch_level(self.mask_ratio)

        # create masked model inputs
        masked_label = label.detach().clone()
        masked_label = masked_label.masked_fill(mask, value=2)

        masked_vx = v_x.detach().clone()
        masked_vx = masked_vx.masked_fill(mask, value=0)

        masked_vy = v_y.detach().clone()
        masked_vy = masked_vy.masked_fill(mask, value=0)

        copy_coorX = coordinates[:, :, 0].detach().clone()
        copy_coorY = coordinates[:, :, 1].detach().clone()

        # create model targets
        target_vx = v_x_ori.masked_fill(~mask, value=0)
        target_vy = v_y_ori.masked_fill(~mask, value=0)

        if 'graph' in self.graph:
            graph = Data()

            if self.graph == 'veno_graph':
                edge_index = create_graph(coordinates, mask, k=self.K)
                edge_index = torch.unique(edge_index.T, dim=0).T
            elif self.graph == 'knn_graph':
                edge_index = knn_graph(coordinates.reshape(-1, 2), k=self.K)
            else:
                c = coordinates.reshape(-1, 2)
                dx = c[0][1] - c[0][0]
                radius = self.K * dx + 0.0001
                edge_index = radius_graph(coordinates.reshape(-1, 2), r=radius)

            # Compute edge_attr
            graph.edge_index = edge_index
            coordinates = coordinates.reshape(-1, 2)
            edge_attr = coordinates[edge_index[1]] - coordinates[edge_index[0]]

            if self.normal:
                edge_distance = torch.norm(edge_attr, dim=1)
                data.edge_distance = edge_distance
                normal_x = edge_attr[:, 0] / data.edge_distance
                normal_y = edge_attr[:, 1] / data.edge_distance
                edge_attr = torch.cat((edge_attr, normal_x, normal_y), dim=-1)

            graph.edge_attr = edge_attr

            graph.x = torch.cat((masked_vx.unsqueeze(-1), masked_vy.unsqueeze(-1), masked_label.unsqueeze(-1),
                                 copy_coorX.unsqueeze(-1), copy_coorY.unsqueeze(-1)),
                                dim=-1).reshape(-1, 5)
            graph.y = torch.cat((target_vx.unsqueeze(-1), target_vy.unsqueeze(-1)), dim=-1).reshape(-1, 2)
            graph.mask = mask.reshape(-1, 1)
            graph.x_e = ext

            return graph
        else:
            return {
                'inp': torch.cat((masked_vx.unsqueeze(-1), masked_vy.unsqueeze(-1), masked_label.unsqueeze(-1),
                                  copy_coorX.unsqueeze(-1), copy_coorY.unsqueeze(-1)),
                                 dim=-1),
                'ext': ext,
                'mask': mask,
                'target': torch.cat((target_vx.unsqueeze(-1), target_vy.unsqueeze(-1)), dim=-1)
            }

    def load(self):
        r"""Load the dataset in memory."""
        print("Loading dataset:", self.path)
        with open(self.path, 'rb') as f:
            self.datas = pickle.load(f)
        f.close()


class Dataset_infer(torch.utils.data.Dataset):
    r"""A base class for representing a Dataset without masking.

    Args:
        path (string): Path to the h5 file.
        transform (callable, optional): A function/transform that takes in a :obj:`graphs4cfd.graph.Graph` object
            and returns a transformed version. The data object will be transformed before every access.
            (default: :obj:`None`)
        training_info (dict, optional): A dictionary containing values of type :obj:`ìnt` for the keys `n_in`,
            `n_out`, `step` and `T`. (default: :obj:`None`)
        idx (int, optional): The index of the simulation to load. If :obj:`None`, then all the simulations are loaded.
            (default: :obj:`None`)
        preload (bool, optional): If :obj:`True`, then the data is loaded in memory. If :obj:`False`, then the data
            is loaded from the h5 file at every access. (default: :obj:`False`)
    """

    def __init__(self, path: str, graph: str, scaler: str, normal: bool, K: int):
        self.path = path
        self.graph = graph
        self.scaler = scaler
        self.normal = normal
        self.K = K
        self.load()

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.datas)

    def __getitem__(self,
                    idx: int):
        r"""Get the idx-th training sample."""
        sample = self.datas[idx]  # [H, W, 5] v_x, v_y, wd, coordinates

        data = sample['x']  # v_x, v_y, label, mask, coordinates
        ext = sample['ext']  # [water_depth, month, weekday, hour]
        ext = ext.reshape(1, -1)

        coordinates = data[:, :, -2:]  # [H, W, 2]

        v_x = data[:, :, 0]
        v_y = data[:, :, 1]

        v_x_ori = v_x.detach().clone()
        v_y_ori = v_y.detach().clone()

        label = 1 - data[:, :, 2]  # 1-boundary, 0-value

        if self.scaler != 'Null':
            if self.scaler == 'minmax':
                v_x = minmax(v_x)
                v_y = minmax(v_y)
            elif self.scaler == 'z':
                v_x = zscore(v_x)
                v_y = zscore(v_y)

        # create matrix to record inference parts
        infer = v_x + label  # to denote the value part
        infer = torch.where(infer != 0, torch.ones_like(infer), torch.zeros_like(infer))  # [obs, boundary:1, target:0]
        mask = ~(infer > 0)  # [obs, boundary:False, target:True]
        label = data[:, :, 2]  # [0-boundary, 1-value]

        # create masked model inputs

        masked_label = label.detach().clone()
        masked_label = masked_label.masked_fill(mask, value=2)  # 0-boundary, 1-value -> 0, 1, 2: infer

        masked_vx = v_x.detach().clone()
        masked_vx = masked_vx.masked_fill(mask, value=0)

        masked_vy = v_y.detach().clone()
        masked_vy = masked_vy.masked_fill(mask, value=0)

        copy_coorX = coordinates[:, :, 0].detach().clone()
        copy_coorY = coordinates[:, :, 1].detach().clone()

        # create model targets

        target_vx = v_x_ori.masked_fill(~mask, value=0)
        target_vy = v_y_ori.masked_fill(~mask, value=0)
        # print(masked_vx.shape)

        if 'graph' in self.graph:
            graph = Data()

            if self.graph == 'veno_graph':
                edge_index = create_graph(coordinates, mask, k=self.K)
                edge_index = torch.unique(edge_index.T, dim=0).T
            elif self.graph == 'knn_graph':
                edge_index = knn_graph(coordinates.reshape(-1, 2), k=self.K)
            else:
                c = coordinates.reshape(-1, 2)
                dx = c[0][1] - c[0][0]
                radius = self.K * dx + 0.0001
                edge_index = radius_graph(coordinates.reshape(-1, 2), r=radius)

            # Compute edge_attr
            graph.edge_index = edge_index
            coordinates = coordinates.reshape(-1, 2)
            edge_attr = coordinates[edge_index[1]] - coordinates[edge_index[0]]

            if self.normal:
                edge_distance = torch.norm(edge_attr, dim=1)
                data.edge_distance = edge_distance
                normal_x = edge_attr[:, 0] / data.edge_distance
                normal_y = edge_attr[:, 1] / data.edge_distance
                edge_attr = torch.cat((edge_attr, normal_x, normal_y), dim=-1)

            graph.edge_attr = edge_attr

            graph.x = torch.cat((masked_vx.unsqueeze(-1), masked_vy.unsqueeze(-1), masked_label.unsqueeze(-1),
                                 copy_coorX.unsqueeze(-1), copy_coorY.unsqueeze(-1)),
                                dim=-1).reshape(-1, 5)
            graph.y = torch.cat((target_vx.unsqueeze(-1), target_vy.unsqueeze(-1)), dim=-1).reshape(-1, 2)
            graph.mask = mask.reshape(-1, 1)
            graph.label = label.reshape(-1, 1)
            graph.x_e = ext

            return graph
        else:
            return {
                'inp': torch.cat((masked_vx.unsqueeze(-1), masked_vy.unsqueeze(-1),
                                  masked_label.unsqueeze(-1),
                                  copy_coorX.unsqueeze(-1), copy_coorY.unsqueeze(-1)),
                                 dim=-1),
                'ext': ext,
                'mask': mask,
                'target': torch.cat((target_vx.unsqueeze(-1), target_vy.unsqueeze(-1)), dim=-1)
            }

    def load(self):
        r"""Load the dataset in memory."""
        print("Loading dataset:", self.path)
        with open(self.path, 'rb') as f:
            self.datas = pickle.load(f)
        f.close()


def loader(dataset, set, config):
    if "graph" in config[set]['graph']:
        from torch_geometric.loader import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=1, shuffle=config[set]['shuffle'],
                                    num_workers=config[set]['num_workers'])
    else:
        from torch.utils.data import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=config[set]['batch_size'], shuffle=config[set]['shuffle'],
                                    num_workers=config[set]['num_workers'])

    return dataset_loader
