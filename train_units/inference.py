from models.pino import PINO_img, PINO_graph
from torch.utils.data import DataLoader
from loss import LpLoss
from datasets.data_utils import Dataset_infer
from tqdm import tqdm
import torch
import yaml
from metrics import rmse, mae, r2_score
import random
import pickle


def get_model(config):
    mode = config['model']['mode']
    nl = config['model']['no_layers']
    if config['train']['graph'] == 'img':
        modes = [mode] * nl
        return PINO_img(modes1=modes, modes2=modes,
                        kernel=config['model']['kernel'], use_exf=bool(config['model']['use_exf']),
                        time=bool(config['model']['time']), in_dim=config['model']['in_dim'],
                        out_dim=config['model']['out_dim'], no_layers=config['model']['no_layers'],
                        act=config['model']['act'], hidden_features=config['model']['hidden_features'],
                        blocks=config['model']['blocks'])
    else:
        modes = [mode] * nl
        return PINO_graph(modes1=modes, modes2=modes,
                          kernel=config['model']['kernel'], use_exf=bool(config['model']['use_exf']),
                          time=bool(config['model']['time']), in_dim=config['model']['in_dim'],
                          out_dim=config['model']['out_dim'], gnn_layers=config['model']['gnn_layers'],
                          no_layers=config['model']['no_layers'], act=config['model']['act'],
                          hidden_features=config['model']['hidden_features'], blocks=config['model']['blocks'])
    # note that, for allday datest, u need train a model without time convolution, i.e., set config['model']['time'] to '0'


def load_model(config, best_epoch, model, device):
    graph = config['train']['graph']
    path = config['train']['save_dir']
    hidden_fea = config['model']['hidden_features']
    gnn_layers = config['model']['gnn_layers']
    no_layers = config['model']['no_layers']
    act = config['model']['act']
    ext = config['model']['use_exf']
    time = config['model']['time']
    mode = config['model']['mode']
    mask_way = config['model']['mask_way']
    kernel = config['model']['kernel']
    mask_ratio = config['train']['mask_ratio']
    K = config['model']['K']
    ckpt_path = f'checkpoints/{path}/neural_operator_h{hidden_fea}_gl{gnn_layers}_nl{no_layers}_{act}_e{ext}_t{time}_K{K}_{kernel}{mode}_{mask_way}_{graph}_{mask_ratio}_{best_epoch}.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    print('Weights loaded from %s' % ckpt_path)
    return model.to(device)


def infer_operator_img(model,
                       data_loader,
                       device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            pred = model(batch['inp'], batch['ext'])
            mask = batch['mask']
            label = batch['inp'][:, 2:3].reshape((-1, 29, 58, 1))
            mask = mask.reshape((-1, 29, 58, 1))
            pred = pred.masked_fill(~mask, value=0)
            pred = pred + batch['inp'][:, 0:2].reshape((-1, 29, 58, 2))
            pred = pred.masked_fill(~(label > 0), value=0)

            predictions.extend(pred)

    return predictions


def infer_operator_graph(model,
                         data_loader,
                         device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch.to(device)
            pred = model(batch)  # [B, H, W, 2]
            label = batch.label
            label = label.reshape((-1, 29, 58, 1))

            mask = batch.mask
            mask = mask.reshape((-1, 29, 58, 1))
            pred = pred.masked_fill(~mask, value=0)
            pred = pred + batch.x[:, 0:2].reshape((-1, 29, 58, 2))
            pred = pred.masked_fill(~(label > 0), value=0)

            predictions.extend(pred)

    return predictions


def loader(dataset, graph):
    if "graph" in graph:
        from torch_geometric.loader import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        from torch.utils.data import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    return dataset_loader


if __name__ == '__main__':
    seed = 42
    print(f'Random seed :{seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    path = '../data/infer_dataset_daytime.pkl'

    with open(path, "rb") as f:
        datas = pickle.load(f)
    f.close()

    id_list = []
    for sample in datas:
        idx = sample['id']
        id_list.append(idx)

    config_file = '../configs/operator/no.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Inference!--------------------------------------------------')

    infer_set = Dataset_infer(path=path, graph='veno_graph', scaler='Null', normal=False, K=2)

    data_loader = loader(infer_set, 'veno_graph')

    print(f'{len(data_loader.dataset)} samples in total!')

    model = get_model(config).to(device)
    model = load_model(config=config, best_epoch=52, model=model, device=device)

    predictions = infer_operator_graph(
        model,
        data_loader,
        device)

    file_path = 'infer_results_daytime.pkl'

    with open(file_path, "wb") as f:
        pickle.dump((id_list, predictions), f)
    f.close()
